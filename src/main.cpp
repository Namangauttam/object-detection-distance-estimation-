// Live YOLOv8 inference on videos with ONNX Runtime (CUDA) + OpenCV, with per-object
// camera-to-object distance (see distance.hpp). Shows each video in a window, saves an
// annotated copy and a per-detection distance CSV.
//
//   yolo_infer [--model PATH] [--source DIR|FILE] [--out DIR] [--conf 0.25] [--iou 0.45]
//              [--camera src/camera.yaml] [--no-show] [--cpu] [--debug-lanes]
//
// Keys in the window: Space = pause, N = next video, Q / Esc = quit.

#include "distance.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

struct Args {
    fs::path model = "model_training/weights/best.onnx";
    fs::path source = "videos";
    fs::path out = "inference_output";
    float conf = 0.25f;
    float iou = 0.45f;
    bool show = true;
    std::string provider = "auto";  // auto | tensorrt | cuda | cpu
    std::string precision = "fp16";  // TensorRT engine precision: fp32 | fp16 | int8 (int8 needs a QDQ model)
    fs::path images, pred_dir;       // evaluation mode: run on an image folder, write YOLO-format predictions
    bool model_default = true;
    fs::path camera = "src/camera.yaml";
    bool debug_lanes = false;
};

static void usage() {
    std::cout << "yolo_infer [--model PATH] [--source DIR|FILE] [--out DIR] [--conf F] [--iou F]"
                 " [--camera YAML] [--provider auto|tensorrt|cuda|cpu] [--precision fp32|fp16|int8] [--cpu]"
                 " [--no-show] [--debug-lanes] [--images DIR --pred-dir DIR]\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (k == "--model") { a.model = next(); a.model_default = false; }
        else if (k == "--source") a.source = next();
        else if (k == "--out") a.out = next();
        else if (k == "--conf") a.conf = std::stof(next());
        else if (k == "--iou") a.iou = std::stof(next());
        else if (k == "--no-show") a.show = false;
        else if (k == "--cpu") a.provider = "cpu";
        else if (k == "--provider") a.provider = next();
        else if (k == "--precision") a.precision = next();
        else if (k == "--images") a.images = next();
        else if (k == "--pred-dir") a.pred_dir = next();
        else if (k == "--camera") a.camera = next();
        else if (k == "--debug-lanes") a.debug_lanes = true;
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else { std::cerr << "Unknown argument: " << k << "\n"; usage(); std::exit(1); }
    }
    return a;
}

static void prepend_path(const char* env_var, const char* fallback) {
#ifdef _WIN32
    std::string dir;
    if (const char* env = std::getenv(env_var)) dir = env;
    if (dir.empty() && fallback) dir = fallback;
    if (dir.empty() || !fs::exists(dir)) return;
    // Read the live process PATH: std::getenv does not see earlier SetEnvironmentVariableA calls.
    std::string path(GetEnvironmentVariableA("PATH", nullptr, 0), '\0');
    if (!path.empty()) path.resize(GetEnvironmentVariableA("PATH", path.data(), (DWORD)path.size()));
    SetEnvironmentVariableA("PATH", (fs::path(dir).make_preferred().string() + ";" + path).c_str());
#endif
}

// Make the CUDA/cuDNN DLLs (from PyTorch) and TensorRT DLLs (from the tensorrt-cu12-libs wheel)
// visible to the ONNX Runtime provider DLLs.
static void add_gpu_dll_dirs() {
#ifdef DEFAULT_CUDA_DLL_DIR
    prepend_path("CUDA_DLL_DIR", DEFAULT_CUDA_DLL_DIR);
#else
    prepend_path("CUDA_DLL_DIR", nullptr);
#endif
#ifdef DEFAULT_TRT_DLL_DIR
    prepend_path("TRT_DLL_DIR", DEFAULT_TRT_DLL_DIR);
#else
    prepend_path("TRT_DLL_DIR", nullptr);
#endif
}

static bool tensorrt_available() {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("nvinfer_10.dll");
    if (h) FreeLibrary(h);
    return h != nullptr;
#else
    return false;
#endif
}

// Ultralytics stores names as a Python dict string: "{0: 'cone', 1: 'person'}".
static std::vector<std::string> parse_names(const std::string& s) {
    std::map<int, std::string> m;
    std::regex re(R"((\d+)\s*:\s*'([^']*)')");
    for (auto it = std::sregex_iterator(s.begin(), s.end(), re); it != std::sregex_iterator(); ++it)
        m[std::stoi((*it)[1])] = (*it)[2];
    std::vector<std::string> names;
    for (auto& [k, v] : m) {
        if (k >= (int)names.size()) names.resize(k + 1);
        names[k] = v;
    }
    return names;
}

static cv::Scalar class_color(int c) {
    static const std::vector<cv::Scalar> pal = {  // BGR
        {26, 122, 255}, {113, 191, 47}, {74, 138, 192}, {230, 230, 230}, {92, 59, 255},
        {255, 140, 79}, {255, 107, 178}, {63, 210, 255}, {214, 196, 46}, {214, 107, 255}};
    return pal[c % pal.size()];
}

static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    return inter / (a.area() + b.area() - inter + 1e-6f);
}

// Greedy per-class NMS.
static std::vector<Detection> nms(std::vector<Detection> dets, float thr) {
    std::sort(dets.begin(), dets.end(), [](auto& x, auto& y) { return x.score > y.score; });
    std::vector<Detection> keep;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!removed[j] && dets[j].cls == dets[i].cls && iou(dets[i].box, dets[j].box) > thr)
                removed[j] = true;
    }
    return keep;
}

enum class Provider { TensorRT, CUDA, CPU };

class YoloOnnx {
public:
    // Tries the requested provider first and falls back TensorRT -> CUDA -> CPU.
    YoloOnnx(const fs::path& model, Provider want, const std::string& trt_precision = "fp16")
        : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"), session_(nullptr), trt_precision_(trt_precision) {
        std::vector<Provider> order;
        if (want == Provider::TensorRT) order = {Provider::TensorRT, Provider::CUDA, Provider::CPU};
        else if (want == Provider::CUDA) order = {Provider::CUDA, Provider::CPU};
        else order = {Provider::CPU};
        for (Provider p : order) {
            try {
                session_ = Ort::Session(env_, model.c_str(), make_options(p, model, trt_precision_));
                provider_ = p == Provider::TensorRT ? "TensorRT" : p == Provider::CUDA ? "CUDA" : "CPU";
                break;
            } catch (const Ort::Exception& e) {
                std::cerr << (p == Provider::TensorRT ? "TensorRT" : p == Provider::CUDA ? "CUDA" : "CPU")
                          << " provider unavailable: " << e.what() << "\n";
                if (p == Provider::CPU) throw;
            }
        }

        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_.GetInputNameAllocated(0, alloc).get();
        out_name_ = session_.GetOutputNameAllocated(0, alloc).get();
        auto in_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto shape = in_info.GetShape();
        size_ = shape.size() == 4 && shape[2] > 0 ? (int)shape[2] : 1280;
        fp16_ = in_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        qdq_ = model.stem().string().find("int8") != std::string::npos;  // QDQ-quantised export (by name)

        auto meta = session_.GetModelMetadata();
        auto names = meta.LookupCustomMetadataMapAllocated("names", alloc);
        if (names) names_ = parse_names(names.get());
    }

    struct Timing { double pre = 0, run = 0, post = 0; };
    const Timing& last_timing() const { return timing_; }

    std::vector<Detection> detect(const cv::Mat& frame, float conf_thr, float iou_thr) {
        using clk = std::chrono::steady_clock;
        auto ms_since = [](clk::time_point t) {
            return std::chrono::duration<double, std::milli>(clk::now() - t).count();
        };
        auto t0 = clk::now();
        // Letterbox to size_ x size_, centred, grey padding (matches Ultralytics).
        float r = std::min((float)size_ / frame.cols, (float)size_ / frame.rows);
        int nw = (int)std::round(frame.cols * r), nh = (int)std::round(frame.rows * r);
        float px = (size_ - nw) / 2.0f, py = (size_ - nh) / 2.0f;
        // The grey canvas is reused; only the resized image area changes between frames.
        if (padded_.empty()) padded_ = cv::Mat(size_, size_, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat roi = padded_(cv::Rect((int)std::round(px - 0.1f), (int)std::round(py - 0.1f), nw, nh));
        cv::resize(frame, roi, roi.size(), 0, 0, cv::INTER_LINEAR);

        // HWC BGR uint8 -> CHW RGB [0,1], converted straight into the tensor buffer (FP32 or FP16).
        const size_t plane = (size_t)size_ * size_;
        const int type = fp16_ ? CV_16F : CV_32F;
        const size_t elem = fp16_ ? sizeof(uint16_t) : sizeof(float);
        input_.resize(3 * plane * elem);
        cv::split(padded_, bgr_);
        for (int c = 0; c < 3; ++c) {  // R -> plane 0, G -> 1, B -> 2
            cv::Mat dst(size_, size_, type, input_.data() + c * plane * elem);
            bgr_[2 - c].convertTo(dst, type, 1.0 / 255.0);
        }

        std::array<int64_t, 4> in_shape{1, 3, size_, size_};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value in = Ort::Value::CreateTensor(
            mem, input_.data(), input_.size(), in_shape.data(), in_shape.size(),
            fp16_ ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        const char* in_names[] = {in_name_.c_str()};
        const char* out_names[] = {out_name_.c_str()};
        timing_.pre = ms_since(t0);
        auto t1 = clk::now();
        auto out = session_.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
        timing_.run = ms_since(t1);
        auto t2 = clk::now();

        // Output: [1, 4 + num_classes, N] with (cx, cy, w, h) in letterboxed pixels.
        auto oshape = out[0].GetTensorTypeAndShapeInfo().GetShape();
        int C = (int)oshape[1], N = (int)oshape[2], nc = C - 4;
        const float* d;
        if (out[0].GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            cv::Mat(1, C * N, CV_16F, (void*)out[0].GetTensorData<uint16_t>()).convertTo(output32_, CV_32F);
            d = output32_.ptr<float>();
        } else {
            d = out[0].GetTensorData<float>();
        }
        if (names_.size() < (size_t)nc)
            for (int c = (int)names_.size(); c < nc; ++c) names_.push_back("class" + std::to_string(c));

        std::vector<Detection> dets;
        for (int i = 0; i < N; ++i) {
            int best = 0;
            float score = d[4 * N + i];
            for (int c = 1; c < nc; ++c) {
                float s = d[(4 + c) * N + i];
                if (s > score) { score = s; best = c; }
            }
            if (score < conf_thr) continue;
            float cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
            float x1 = std::clamp((cx - w / 2 - px) / r, 0.f, (float)frame.cols);
            float y1 = std::clamp((cy - h / 2 - py) / r, 0.f, (float)frame.rows);
            float x2 = std::clamp((cx + w / 2 - px) / r, 0.f, (float)frame.cols);
            float y2 = std::clamp((cy + h / 2 - py) / r, 0.f, (float)frame.rows);
            if (x2 - x1 < 1 || y2 - y1 < 1) continue;
            dets.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), score, best});
        }
        auto kept = nms(std::move(dets), iou_thr);
        timing_.post = ms_since(t2);
        return kept;
    }

    const std::vector<std::string>& names() const { return names_; }
    std::string provider() const {
        if (provider_ == "TensorRT") {
            std::string p = trt_precision_;
            std::transform(p.begin(), p.end(), p.begin(), ::toupper);
            return "TensorRT " + p;
        }
        return provider_ + (fp16_ ? " FP16" : qdq_ ? " INT8" : " FP32");
    }
    int size() const { return size_; }

private:
    Ort::Env env_;
    Ort::Session session_;
    static Ort::SessionOptions make_options(Provider p, const fs::path& model, const std::string& precision) {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (p == Provider::TensorRT) {
            // Engine + timing caches next to the model: the first run builds (minutes), later runs load.
            static std::string cache = (model.parent_path() / "trt_cache").string();
            fs::create_directories(cache);
            bool cached = false;
            for (auto& e : fs::directory_iterator(cache)) cached |= e.path().extension() == ".engine";
            if (!cached) std::cout << "Building TensorRT engine (first run only, may take a few minutes)...\n";
            const OrtApi& api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2* trt = nullptr;
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
            const char* keys[] = {"device_id", "trt_fp16_enable", "trt_engine_cache_enable",
                                  "trt_engine_cache_path", "trt_timing_cache_enable", "trt_timing_cache_path",
                                  "trt_max_workspace_size", "trt_builder_optimization_level", "trt_int8_enable"};
            const char* fp16 = precision == "fp32" ? "0" : "1";  // int8 keeps fp16 for non-quantised layers
            const char* int8 = precision == "int8" ? "1" : "0";  // explicit QDQ quantisation, no calibration table
            const char* vals[] = {"0", fp16, "1", cache.c_str(), "1", cache.c_str(), "2147483648", "3", int8};
            OrtStatus* st = api.UpdateTensorRTProviderOptions(trt, keys, vals, sizeof(keys) / sizeof(keys[0]));
            if (st) { api.ReleaseTensorRTProviderOptions(trt); Ort::ThrowOnError(st); }
            try {
                opts.AppendExecutionProvider_TensorRT_V2(*trt);
            } catch (...) {
                api.ReleaseTensorRTProviderOptions(trt);
                throw;
            }
            api.ReleaseTensorRTProviderOptions(trt);
        }
        if (p != Provider::CPU) {  // CUDA also runs any nodes TensorRT doesn't support
            OrtCUDAProviderOptions cuda{};
            cuda.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda);
        }
        return opts;
    }

    std::string in_name_, out_name_, provider_ = "CPU";
    std::vector<std::string> names_;
    std::vector<uint8_t> input_;  // raw tensor bytes (FP32 or FP16)
    cv::Mat padded_, output32_;
    std::vector<cv::Mat> bgr_;
    bool fp16_ = false, qdq_ = false;
    std::string trt_precision_;
    Timing timing_;
    int size_ = 1280;
};

static void draw(cv::Mat& img, const std::vector<Detection>& dets, const std::vector<DistanceEstimate>& est,
                 const std::vector<std::string>& names) {
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        const auto& e = est[i];
        cv::Scalar col = class_color(d.cls);
        cv::Rect box(d.box);
        cv::rectangle(img, box, col, 2, cv::LINE_AA);
        char label[96];
        if (e.smoothed > 0)  // "~" = bottom hidden by the bonnet, size-only estimate
            std::snprintf(label, sizeof(label), "%s %.2f | %s%.1f m", names[d.cls].c_str(), d.score,
                          e.bonnet_cut ? "~" : "", e.smoothed);
        else
            std::snprintf(label, sizeof(label), "%s %.2f", names[d.cls].c_str(), d.score);
        int base = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        int ty = std::max(box.y, ts.height + 6);
        cv::rectangle(img, {box.x, ty - ts.height - 6, ts.width + 6, ts.height + 6}, col, cv::FILLED);
        cv::putText(img, label, {box.x + 3, ty - 4}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
    }
}

static void draw_hud(cv::Mat& img, const std::string& text, int row = 0) {
    int base = 0;
    cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base);
    int h = ts.height + 16, y0 = row * h;
    cv::Mat roi = img(cv::Rect(0, y0, std::min(img.cols, ts.width + 20), h));
    cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.4, dark, 0.6, 0, roi);
    cv::putText(img, text, {10, y0 + ts.height + 8}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {255, 255, 255}, 1,
                cv::LINE_AA);
}

static void draw_horizon(cv::Mat& img, const HorizonEstimator& hz, bool lanes) {
    if (lanes)
        for (auto& l : hz.lane_lines())
            cv::line(img, {(int)l[0], (int)l[1]}, {(int)l[2], (int)l[3]}, {255, 0, 255}, 1, cv::LINE_AA);
    int y = (int)std::round(hz.y());
    if (y < 0 || y >= img.rows) return;
    cv::Scalar col = hz.locked() ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 165, 255);
    for (int x = 0; x < img.cols; x += 24) cv::line(img, {x, y}, {std::min(x + 12, img.cols - 1), y}, col, 1);
}

static std::vector<fs::path> list_videos(const fs::path& src) {
    std::vector<fs::path> v;
    if (fs::is_regular_file(src)) return {src};
    if (!fs::is_directory(src)) return v;
    for (auto& e : fs::directory_iterator(src)) {
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") v.push_back(e.path());
    }
    std::sort(v.begin(), v.end());
    return v;
}

// Median / p95 of per-frame times, skipping warm-up frames (GPU clocks, lazy allocations).
static std::string timing_summary(std::vector<double> total, std::vector<double> run, size_t warmup = 10) {
    if (total.size() <= warmup + 1) warmup = 0;
    total.erase(total.begin(), total.begin() + warmup);
    run.erase(run.begin(), run.begin() + warmup);
    if (total.empty()) return "no frames";
    auto pct = [](std::vector<double> v, double q) {
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
    };
    char buf[200];
    std::snprintf(buf, sizeof(buf), "median total %.2f ms, p95 %.2f ms, median model %.2f ms (%zu timed frames)",
                  pct(total, 0.5), pct(total, 0.95), pct(run, 0.5), total.size());
    return buf;
}

// Evaluation mode: predictions for every image in a folder, one YOLO txt per image
// ("cls cx cy w h conf", normalised), for computing mAP against the label files.
static int run_images(YoloOnnx& model, const Args& args) {
    std::vector<fs::path> imgs;
    for (auto& e : fs::directory_iterator(args.images)) {
        auto ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") imgs.push_back(e.path());
    }
    std::sort(imgs.begin(), imgs.end());
    if (imgs.empty()) { std::cerr << "No images in " << args.images << "\n"; return 1; }
    fs::create_directories(args.pred_dir);
    std::vector<double> total, run;
    for (auto& p : imgs) {
        cv::Mat img = cv::imread(p.string());
        if (img.empty()) continue;
        auto t0 = std::chrono::steady_clock::now();
        auto dets = model.detect(img, args.conf, args.iou);
        total.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        run.push_back(model.last_timing().run);
        std::ofstream f(args.pred_dir / (p.stem().string() + ".txt"));
        for (auto& d : dets)
            f << d.cls << ' ' << (d.box.x + d.box.width / 2) / img.cols << ' ' << (d.box.y + d.box.height / 2) / img.rows
              << ' ' << d.box.width / img.cols << ' ' << d.box.height / img.rows << ' ' << d.score << '\n';
    }
    std::cout << "EVAL images " << total.size() << " -> " << args.pred_dir.string() << "  "
              << timing_summary(total, run) << "\n";
    return 0;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    add_gpu_dll_dirs();

    Provider provider;
    if (args.provider == "tensorrt" || args.provider == "trt") provider = Provider::TensorRT;
    else if (args.provider == "cuda") provider = Provider::CUDA;
    else if (args.provider == "cpu") provider = Provider::CPU;
    else if (args.provider == "auto") provider = tensorrt_available() ? Provider::TensorRT : Provider::CUDA;
    else { std::cerr << "Unknown --provider " << args.provider << "\n"; return 1; }

    // TensorRT builds its own FP16 engine from the FP32 model; plain CUDA is faster with the FP16 export.
    if (args.model_default && provider == Provider::CUDA) {
        fs::path fp16 = args.model.parent_path() / (args.model.stem().string() + "_fp16.onnx");
        if (fs::exists(fp16)) args.model = fp16;
    }
    if (!fs::exists(args.model)) {
        std::cerr << "Model not found: " << args.model
                  << "\nExport it first: python src/export_onnx.py\n";
        return 1;
    }
    if (args.precision != "fp32" && args.precision != "fp16" && args.precision != "int8") {
        std::cerr << "Unknown --precision " << args.precision << "\n";
        return 1;
    }
    std::vector<fs::path> videos;
    if (args.images.empty()) {
        videos = list_videos(args.source);
        if (videos.empty()) {
            std::cerr << "No videos found in " << args.source << "\n";
            return 1;
        }
        fs::create_directories(args.out);
    } else if (args.pred_dir.empty()) {
        std::cerr << "--images needs --pred-dir\n";
        return 1;
    }

    auto t_load = std::chrono::steady_clock::now();
    YoloOnnx model(args.model, provider, args.precision);
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
    std::cout << "Model: " << args.model.string() << "  input " << model.size() << "x" << model.size()
              << "  provider " << model.provider() << "  session ready in " << load_s << " s\nClasses:";
    for (auto& n : model.names()) std::cout << " " << n;
    std::cout << "\n";
    if (!args.images.empty()) return run_images(model, args);

    bool cam_found = false;
    CameraConfig cam = CameraConfig::load(args.camera.string(), &cam_found);
    std::cout << "Camera config: " << (cam_found ? args.camera.string() : std::string("not found, using defaults"))
              << "\n";

    const std::string win = "YOLO inference  (Space pause | N next | Q quit)";
    if (args.show) {
        cv::namedWindow(win, cv::WINDOW_NORMAL);
        cv::resizeWindow(win, 1280, 720);
    }

    bool quit = false;
    for (const auto& vpath : videos) {
        if (quit) break;
        cv::VideoCapture cap(vpath.string());
        if (!cap.isOpened()) { std::cerr << "Cannot open " << vpath << "\n"; continue; }
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30;
        int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        cv::Size sz((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        fs::path out_path = args.out / (vpath.stem().string() + "_inference.mp4");
        cv::VideoWriter writer(out_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, sz);
        if (!writer.isOpened()) std::cerr << "Warning: cannot write " << out_path << "\n";

        // Distance pipeline is re-initialised per video (resolution / mounting may differ).
        cam.finalize(sz);
        HorizonEstimator horizon(cam);
        DistanceEstimator dist(cam);
        Tracker tracker;
        fs::path csv_path = args.out / (vpath.stem().string() + "_distances.csv");
        std::ofstream csv(csv_path);
        csv << "frame,track_id,class,conf,x1,y1,x2,y2,dist_ground_m,dist_size_m,dist_fused_m,sigma_m,"
               "dist_smoothed_m,bonnet_cut,horizon_y,pitch_deg,camera_height_m\n";

        std::map<std::string, int> counts;
        double infer_ms_sum = 0, pre_sum = 0, run_sum = 0, post_sum = 0;
        std::vector<double> frame_ms, model_ms;
        int n = 0;
        bool paused = false;
        cv::Mat frame, shown;
        auto t_prev = std::chrono::steady_clock::now();
        double disp_fps = 0;

        while (true) {
            if (!paused) {
                if (!cap.read(frame) || frame.empty()) break;
                auto t0 = std::chrono::steady_clock::now();
                auto dets = model.detect(frame, args.conf, args.iou);
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                infer_ms_sum += ms;
                const auto& tm = model.last_timing();
                pre_sum += tm.pre; run_sum += tm.run; post_sum += tm.post;
                frame_ms.push_back(ms);
                model_ms.push_back(tm.run);
                ++n;
                for (auto& d : dets) counts[model.names()[d.cls]]++;

                horizon.update(frame);
                std::vector<DistanceEstimate> est;
                for (auto& d : dets) est.push_back(dist.estimate(d, model.names()[d.cls], horizon));
                tracker.update(dets, est);
                for (size_t i = 0; i < dets.size(); ++i) {
                    const auto& d = dets[i];
                    const auto& e = est[i];
                    csv << n << ',' << e.track_id << ',' << model.names()[d.cls] << ',' << d.score << ','
                        << d.box.x << ',' << d.box.y << ',' << d.box.x + d.box.width << ','
                        << d.box.y + d.box.height << ',' << e.ground << ',' << e.size << ',' << e.fused << ','
                        << e.sigma << ',' << e.smoothed << ',' << e.bonnet_cut << ',' << horizon.y() << ','
                        << horizon.pitch_deg() << ',' << dist.camera_height() << '\n';
                }

                shown = frame.clone();
                draw_horizon(shown, horizon, args.debug_lanes);
                draw(shown, dets, est, model.names());
                auto now = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double>(now - t_prev).count();
                t_prev = now;
                disp_fps = disp_fps == 0 ? 1.0 / dt : 0.9 * disp_fps + 0.1 / dt;
                char hud[256];
                std::snprintf(hud, sizeof(hud), "%s  %d/%d  |  %s  infer %.1f ms  |  %.1f FPS  |  %zu obj",
                              vpath.stem().string().c_str(), n, total, model.provider().c_str(), ms, disp_fps,
                              dets.size());
                draw_hud(shown, hud);
                std::snprintf(hud, sizeof(hud), "horizon %s  pitch %+.1f deg  |  camera height %.2f m (%s)",
                              cam.auto_horizon ? (horizon.locked() ? "tracking" : "searching") : "fixed",
                              horizon.pitch_deg(), dist.camera_height(),
                              !cam.auto_camera_height ? "fixed"
                              : dist.height_samples() >= 15 ? "auto"
                                                            : "config");
                draw_hud(shown, hud, 1);
                if (writer.isOpened()) writer.write(shown);
            }
            if (args.show) {
                cv::imshow(win, shown);
                int k = cv::waitKey(paused ? 30 : 1) & 0xFF;
                if (k == 'q' || k == 'Q' || k == 27) { quit = true; break; }
                if (k == 'n' || k == 'N') break;
                if (k == ' ') paused = !paused;
                if (cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) < 1) { quit = true; break; }
            }
        }
        writer.release();
        std::cout << vpath.filename().string() << " -> " << out_path.string() << "  frames " << n
                  << "  avg ms: total " << (n ? infer_ms_sum / n : 0) << " (pre " << (n ? pre_sum / n : 0)
                  << ", model " << (n ? run_sum / n : 0) << ", post " << (n ? post_sum / n : 0)
                  << ")  detections {";
        for (auto& [k, v] : counts) std::cout << " " << k << ":" << v;
        std::cout << " }\n    timing: " << timing_summary(frame_ms, model_ms) << "\n    distance: horizon " << (horizon.locked() ? "tracked" : "NOT locked") << " (pitch "
                  << horizon.pitch_deg() << " deg), camera height " << dist.camera_height() << " m from "
                  << dist.height_samples() << " samples  -> " << csv_path.string() << "\n";
    }
    if (args.show) cv::destroyAllWindows();
    return 0;
}
