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
    bool cpu = false;
    bool model_default = true;
    fs::path camera = "src/camera.yaml";
    bool debug_lanes = false;
};

static void usage() {
    std::cout << "yolo_infer [--model PATH] [--source DIR|FILE] [--out DIR] [--conf F] [--iou F]"
                 " [--camera YAML] [--no-show] [--cpu] [--debug-lanes]\n";
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
        else if (k == "--cpu") a.cpu = true;
        else if (k == "--camera") a.camera = next();
        else if (k == "--debug-lanes") a.debug_lanes = true;
        else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else { std::cerr << "Unknown argument: " << k << "\n"; usage(); std::exit(1); }
    }
    return a;
}

// Make the CUDA/cuDNN DLLs shipped with PyTorch visible to onnxruntime_providers_cuda.dll.
static void add_cuda_dll_dir() {
#ifdef _WIN32
    std::string dir;
    if (const char* env = std::getenv("CUDA_DLL_DIR")) dir = env;
#ifdef DEFAULT_CUDA_DLL_DIR
    if (dir.empty()) dir = DEFAULT_CUDA_DLL_DIR;
#endif
    if (dir.empty() || !fs::exists(dir)) return;
    std::string path = std::getenv("PATH") ? std::getenv("PATH") : "";
    SetEnvironmentVariableA("PATH", (fs::path(dir).make_preferred().string() + ";" + path).c_str());
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

class YoloOnnx {
public:
    YoloOnnx(const fs::path& model, bool cpu)
        : env_(ORT_LOGGING_LEVEL_WARNING, "yolo"), session_(nullptr) {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (!cpu) {
            try {
                OrtCUDAProviderOptions cuda{};
                cuda.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda);
                provider_ = "CUDA";
            } catch (const Ort::Exception& e) {
                std::cerr << "CUDA provider unavailable (" << e.what() << "), using CPU.\n";
            }
        }
        session_ = Ort::Session(env_, model.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;
        in_name_ = session_.GetInputNameAllocated(0, alloc).get();
        out_name_ = session_.GetOutputNameAllocated(0, alloc).get();
        auto in_info = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        auto shape = in_info.GetShape();
        size_ = shape.size() == 4 && shape[2] > 0 ? (int)shape[2] : 1280;
        fp16_ = in_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

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
    std::string provider() const { return provider_ + (fp16_ ? " FP16" : " FP32"); }
    int size() const { return size_; }

private:
    Ort::Env env_;
    Ort::Session session_;
    std::string in_name_, out_name_, provider_ = "CPU";
    std::vector<std::string> names_;
    std::vector<uint8_t> input_;  // raw tensor bytes (FP32 or FP16)
    cv::Mat padded_, output32_;
    std::vector<cv::Mat> bgr_;
    bool fp16_ = false;
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

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.model_default && !args.cpu) {  // prefer the faster FP16 export on GPU
        fs::path fp16 = args.model.parent_path() / (args.model.stem().string() + "_fp16.onnx");
        if (fs::exists(fp16)) args.model = fp16;
    }
    if (!fs::exists(args.model)) {
        std::cerr << "Model not found: " << args.model
                  << "\nExport it first: python src/export_onnx.py\n";
        return 1;
    }
    auto videos = list_videos(args.source);
    if (videos.empty()) {
        std::cerr << "No videos found in " << args.source << "\n";
        return 1;
    }
    fs::create_directories(args.out);
    add_cuda_dll_dir();

    YoloOnnx model(args.model, args.cpu);
    std::cout << "Model: " << args.model.string() << "  input " << model.size() << "x" << model.size()
              << "  provider " << model.provider() << "\nClasses:";
    for (auto& n : model.names()) std::cout << " " << n;
    std::cout << "\n";

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
        std::cout << " }\n    distance: horizon " << (horizon.locked() ? "tracked" : "NOT locked") << " (pitch "
                  << horizon.pitch_deg() << " deg), camera height " << dist.camera_height() << " m from "
                  << dist.height_samples() << " samples  -> " << csv_path.string() << "\n";
    }
    if (args.show) cv::destroyAllWindows();
    return 0;
}
