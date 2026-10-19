// Lens calibration from a checkerboard, writing the result into camera.yaml.
//
//   calibrate_camera --make-board board.png
//       Writes a printable 10x7-square board (9x6 inner corners, 25 mm squares at 300 dpi).
//
//   calibrate_camera --source calib.mp4|DIR [--board 9x6] [--square 0.025] [--out src/camera.yaml]
//                    [--max-views 40] [--show]
//       Finds the board in the video/images, calibrates focal length + lens distortion,
//       updates fx/fy/cx/cy/dist in the YAML (other settings are kept) and saves
//       calibration_preview.jpg (original | lens-corrected) next to it.

#if __has_include(<opencv2/calib.hpp>)  // OpenCV 5
#include <opencv2/calib.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/objdetect.hpp>
#else  // OpenCV 4
#include <opencv2/calib3d.hpp>
#endif
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    fs::path source, out = "src/camera.yaml", make_board;
    cv::Size board{9, 6};  // inner corners (cols x rows)
    double square = 0.025;  // metres
    int max_views = 40;
    bool show = false;
};

static void usage() {
    std::cout << "calibrate_camera --make-board board.png\n"
                 "calibrate_camera --source VIDEO|DIR [--board 9x6] [--square 0.025] [--out src/camera.yaml]"
                 " [--max-views 40] [--show]\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (k == "--source") a.source = next();
        else if (k == "--out") a.out = next();
        else if (k == "--make-board") a.make_board = next();
        else if (k == "--square") a.square = std::stod(next());
        else if (k == "--max-views") a.max_views = std::stoi(next());
        else if (k == "--show") a.show = true;
        else if (k == "--board") {
            std::string v = next();
            int c = 0, r = 0;
            if (std::sscanf(v.c_str(), "%dx%d", &c, &r) != 2 || c < 3 || r < 3) {
                std::cerr << "--board must look like 9x6 (inner corners)\n";
                std::exit(1);
            }
            a.board = {c, r};
        } else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
        else { std::cerr << "Unknown argument: " << k << "\n"; usage(); std::exit(1); }
    }
    return a;
}

static int make_board(const Args& a) {
    const int px = (int)std::round(300.0 * 25.0 / 25.4);  // 25 mm at 300 dpi
    const int cols = a.board.width + 1, rows = a.board.height + 1, margin = px;
    cv::Mat img(rows * px + 2 * margin, cols * px + 2 * margin, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if ((r + c) % 2 == 0) img(cv::Rect(margin + c * px, margin + r * px, px, px)).setTo(0);
    cv::imwrite(a.make_board.string(), img);
    std::cout << "Wrote " << a.make_board << " (" << cols << "x" << rows << " squares, " << a.board.width << "x"
              << a.board.height << " inner corners)\n"
              << "Print at 100% / 'actual size' (no fit-to-page), tape it flat to something rigid,\n"
              << "and measure one square with a ruler (pass it as --square in metres, e.g. 0.025).\n";
    return 0;
}

// Replace (or append) "key: value" lines, keeping every other line and comment in the YAML.
static void update_yaml(const fs::path& path, const std::vector<std::pair<std::string, std::string>>& kv) {
    std::vector<std::string> lines;
    if (std::ifstream in(path); in) {
        for (std::string l; std::getline(in, l);) lines.push_back(l);
    } else {
        lines = {"%YAML:1.0", "---"};
    }
    for (auto& [key, value] : kv) {
        std::regex re("^" + key + "\\s*:.*$");
        bool done = false;
        for (auto& l : lines)
            if (std::regex_match(l, re)) { l = key + ": " + value; done = true; break; }
        if (!done) lines.push_back(key + ": " + value);
    }
    std::ofstream out(path);
    for (auto& l : lines) out << l << "\n";
}

static std::string lower_ext(const fs::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}
static bool is_image(const fs::path& p) {
    auto e = lower_ext(p);
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp";
}
static bool is_video(const fs::path& p) {
    auto e = lower_ext(p);
    return e == ".mp4" || e == ".avi" || e == ".mov" || e == ".mkv";
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (!a.make_board.empty()) return make_board(a);
    if (a.source.empty()) { usage(); return 1; }

    // --source may be one video, one image folder, or a folder of videos (and/or images).
    std::vector<fs::path> videos, images;
    if (fs::is_directory(a.source)) {
        for (auto& e : fs::directory_iterator(a.source)) {
            if (is_video(e.path())) videos.push_back(e.path());
            else if (is_image(e.path())) images.push_back(e.path());
        }
        std::sort(videos.begin(), videos.end());
        std::sort(images.begin(), images.end());
    } else if (fs::is_regular_file(a.source)) {
        (is_image(a.source) ? images : videos).push_back(a.source);
    }
    if (videos.empty() && images.empty()) { std::cerr << "No videos or images found in " << a.source << "\n"; return 1; }

    std::vector<std::vector<cv::Point2f>> img_pts;
    cv::Mat sample, gray;
    cv::Size size;
    const int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK;
    int tried = 0;
    bool stop = false;
    const std::string win = "calibration (found corners)";

    // Returns after adding the frame's board corners if they differ enough from views already kept.
    auto process = [&](const cv::Mat& frame) {
        if (frame.empty()) return;
        if (size.empty()) size = frame.size();
        if (frame.size() != size) { std::cerr << "Skipping frame with different resolution\n"; return; }
        ++tried;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        if (!cv::findChessboardCorners(gray, a.board, corners, flags)) return;
        cv::cornerSubPix(gray, corners, {11, 11}, {-1, -1},
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
        double min_diff = 1e9;  // keep only views that differ in position / size / tilt
        for (auto& v : img_pts) {
            double s = 0;
            for (size_t i = 0; i < v.size(); ++i) s += cv::norm(v[i] - corners[i]);
            min_diff = std::min(min_diff, s / v.size());
        }
        if (min_diff < 0.04 * size.width) return;
        img_pts.push_back(corners);
        if (sample.empty()) sample = frame.clone();
        if (a.show) {
            cv::Mat vis = frame.clone();
            cv::drawChessboardCorners(vis, a.board, corners, true);
            cv::imshow(win, vis);
            if ((cv::waitKey(1) & 0xFF) == 'q') stop = true;
        }
    };

    std::cout << "Board: " << a.board.width << "x" << a.board.height << " inner corners, " << a.square * 1000
              << " mm squares\n";
    for (auto& v : videos) {
        if (stop) break;
        cv::VideoCapture cap(v.string());
        if (!cap.isOpened()) { std::cerr << "Cannot open " << v << "\n"; continue; }
        int total = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        int step = std::max(1, total / 200);  // ~200 candidate frames per video
        size_t before = img_pts.size();
        cv::Mat frame;
        for (int i = 0; !stop && cap.read(frame); ++i)
            if (i % step == 0) process(frame);
        std::cout << "  " << v.filename().string() << ": " << img_pts.size() - before << " views\n";
    }
    if (!images.empty()) {
        size_t before = img_pts.size();
        for (auto& p : images) {
            if (stop) break;
            process(cv::imread(p.string()));
        }
        std::cout << "  " << images.size() << " images: " << img_pts.size() - before << " views\n";
    }
    if (a.show) cv::destroyAllWindows();
    std::cout << "Board found in " << img_pts.size() << " distinct views (" << tried << " frames checked)\n";

    // Many videos can give many views; keep an even spread of at most --max-views.
    if ((int)img_pts.size() > a.max_views) {
        std::vector<std::vector<cv::Point2f>> spread;
        for (int i = 0; i < a.max_views; ++i) spread.push_back(img_pts[(size_t)i * img_pts.size() / a.max_views]);
        img_pts.swap(spread);
        std::cout << "Using " << img_pts.size() << " evenly spread views\n";
    }
    if (img_pts.size() < 10) {
        std::cerr << "Need at least 10 views. Move the board around the whole image, tilt it, vary the "
                     "distance, keep it sharp and fully visible, and check --board matches the INNER corners.\n";
        return 1;
    }

    // Coverage check: lens distortion is strongest at the edges, so corners must reach them.
    int covered = 0;
    bool cell[4][4] = {};
    for (auto& v : img_pts)
        for (auto& p : v) cell[std::clamp((int)(4 * p.y / size.height), 0, 3)][std::clamp((int)(4 * p.x / size.width), 0, 3)] = true;
    for (auto& r : cell)
        for (bool c : r) covered += c;

    std::vector<cv::Point3f> grid;
    for (int r = 0; r < a.board.height; ++r)
        for (int c = 0; c < a.board.width; ++c) grid.push_back(cv::Point3f(c * a.square, r * a.square, 0));

    auto calibrate = [&](std::vector<std::vector<cv::Point2f>>& views, cv::Mat& K, cv::Mat& D,
                         std::vector<double>& per_view) {
        std::vector<std::vector<cv::Point3f>> obj(views.size(), grid);
        std::vector<cv::Mat> rv, tv;
        double rms = cv::calibrateCamera(obj, views, size, K, D, rv, tv);
        per_view.clear();
        for (size_t i = 0; i < views.size(); ++i) {
            std::vector<cv::Point2f> proj;
            cv::projectPoints(grid, rv[i], tv[i], K, D, proj);
            per_view.push_back(cv::norm(views[i], proj, cv::NORM_L2) / std::sqrt((double)proj.size()));
        }
        return rms;
    };

    cv::Mat K, D;
    std::vector<double> err;
    double rms = calibrate(img_pts, K, D, err);
    // One refinement pass: drop blurry / mis-detected views (> 3x median error).
    std::vector<double> sorted = err;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[sorted.size() / 2];
    std::vector<std::vector<cv::Point2f>> kept;
    for (size_t i = 0; i < img_pts.size(); ++i)
        if (err[i] <= 3 * med) kept.push_back(img_pts[i]);
    if (kept.size() >= 10 && kept.size() < img_pts.size()) {
        std::cout << "Dropped " << img_pts.size() - kept.size() << " outlier view(s), recalibrating\n";
        img_pts = kept;
        rms = calibrate(img_pts, K, D, err);
    }

    double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
    double hfov = 2 * std::atan(size.width / (2 * fx)) * 180 / CV_PI;
    double vfov = 2 * std::atan(size.height / (2 * fy)) * 180 / CV_PI;
    std::vector<double> d(D.begin<double>(), D.end<double>());

    std::cout << "\nResult (" << size.width << "x" << size.height << ", " << img_pts.size() << " views)\n"
              << "  RMS reprojection error: " << rms << " px  "
              << (rms < 0.5 ? "(good)" : rms < 1.0 ? "(acceptable)" : "(poor - recapture with sharper views)")
              << "\n  fx " << fx << "  fy " << fy << "  cx " << cx << "  cy " << cy << "\n  field of view "
              << hfov << " x " << vfov << " deg\n  distortion k1 k2 p1 p2 k3:";
    for (double v : d) std::cout << " " << v;
    std::cout << "\n  image coverage " << covered << "/16 cells"
              << (covered < 14 ? "  WARNING: move the board into the corners/edges too" : "") << "\n";

    char date[16];
    std::time_t now = std::time(nullptr);
    std::strftime(date, sizeof(date), "%d/%m/%Y", std::localtime(&now));
    std::ostringstream note, dist;
    note << "  # calibrated " << date << ", RMS " << std::fixed << std::setprecision(3) << rms << " px, "
         << img_pts.size() << " views";
    dist << std::setprecision(8) << "[";
    for (size_t i = 0; i < d.size(); ++i) dist << (i ? ", " : "") << d[i];
    dist << "]";
    auto f = [](double v) { std::ostringstream s; s << std::fixed << std::setprecision(4) << v; return s.str(); };
    update_yaml(a.out, {{"fx", f(fx) + note.str()},
                        {"fy", f(fy)},
                        {"cx", f(cx)},
                        {"cy", f(cy)},
                        {"dist", dist.str()},
                        {"calib_width", std::to_string(size.width)},
                        {"calib_height", std::to_string(size.height)},
                        {"hfov_deg", f(hfov) + "  # from calibration (only used if fx is 0)"}});
    std::cout << "Updated " << a.out << "\n";

    // Before/after preview so you can see the lens correction.
    cv::Mat fixed, preview;
    cv::undistort(sample, fixed, K, D);
    cv::putText(sample, "original", {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.2, {0, 255, 255}, 2);
    cv::putText(fixed, "lens corrected", {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.2, {0, 255, 255}, 2);
    cv::hconcat(sample, fixed, preview);
    fs::path prev = a.out.parent_path() / "calibration_preview.jpg";
    cv::imwrite(prev.string(), preview);
    std::cout << "Preview: " << prev << "\n";
    return 0;
}
