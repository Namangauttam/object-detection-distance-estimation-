#include "distance.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>

static constexpr double kPi = 3.14159265358979323846;
static double deg2rad(double d) { return d * kPi / 180.0; }
static double rad2deg(double r) { return r * 180.0 / kPi; }

// ---------------------------------------------------------------- CameraConfig

CameraConfig CameraConfig::load(const std::string& path, bool* found) {
    CameraConfig c;
    c.sizes = {{"cone", {0.45, 0.05}}, {"person", {1.70, 0.10}}, {"chair", {0.40, 0.25}},
               {"house", {0.30, 0.25}}, {"barrier", {1.00, 0.20}}};
    cv::FileStorage fs;
    try {
        fs.open(path, cv::FileStorage::READ);
    } catch (const cv::Exception&) {
    }
    if (found) *found = fs.isOpened();
    if (!fs.isOpened()) return c;

    auto num = [&](const char* key, double& v) {
        if (!fs[key].empty()) v = (double)fs[key];
    };
    auto flag = [&](const char* key, bool& v) {
        if (!fs[key].empty()) v = (int)fs[key] != 0;
    };
    num("hfov_deg", c.hfov_deg);
    num("fx", c.fx);
    num("fy", c.fy);
    num("cx", c.cx);
    num("cy", c.cy);
    num("camera_height_m", c.camera_height_m);
    flag("auto_camera_height", c.auto_camera_height);
    num("pitch_deg", c.pitch_deg);
    flag("auto_horizon", c.auto_horizon);
    num("bonnet_top_frac", c.bonnet_top_frac);
    num("max_distance_m", c.max_distance_m);
    if (!fs["calib_width"].empty()) c.calib_width = (int)fs["calib_width"];
    if (!fs["calib_height"].empty()) c.calib_height = (int)fs["calib_height"];
    if (fs["dist"].isSeq()) {
        c.dist.clear();
        for (auto it = fs["dist"].begin(); it != fs["dist"].end(); ++it) c.dist.push_back((double)*it);
    }
    cv::FileNode sz = fs["object_heights"];
    if (sz.isMap()) {
        for (auto it = sz.begin(); it != sz.end(); ++it) {
            cv::FileNode n = *it;
            if (n.isSeq() && n.size() >= 1)
                c.sizes[n.name()] = {(double)n[0], n.size() >= 2 ? (double)n[1] : 0.15};
        }
    }
    return c;
}

void CameraConfig::finalize(cv::Size image_size) {
    // Calibrated at another resolution (same sensor crop assumed): rescale the intrinsics once.
    if (fx > 0 && calib_width > 0 && calib_height > 0 &&
        (calib_width != image_size.width || calib_height != image_size.height)) {
        double sx = (double)image_size.width / calib_width, sy = (double)image_size.height / calib_height;
        fx *= sx; cx *= sx; fy *= sy; cy *= sy;
        std::cerr << "Note: intrinsics calibrated at " << calib_width << "x" << calib_height << ", rescaled to "
                  << image_size.width << "x" << image_size.height << "\n";
        calib_width = image_size.width;
        calib_height = image_size.height;
    }
    image = image_size;
    if (fx <= 0) {
        fx = (image.width / 2.0) / std::tan(deg2rad(hfov_deg) / 2.0);
        fy = fx;  // square pixels
    }
    if (fy <= 0) fy = fx;
    if (cx <= 0) cx = image.width / 2.0;
    if (cy <= 0) cy = image.height / 2.0;
}

cv::Point2d CameraConfig::normalize(cv::Point2d px) const {
    double x = (px.x - cx) / fx, y = (px.y - cy) / fy;
    if (dist.size() < 4) return {x, y};
    // Invert Brown-Conrady distortion by fixed-point iteration.
    double k1 = dist[0], k2 = dist[1], p1 = dist[2], p2 = dist[3], k3 = dist.size() > 4 ? dist[4] : 0;
    double x0 = x, y0 = y;
    for (int i = 0; i < 10; ++i) {
        double r2 = x * x + y * y;
        double radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        double dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        double dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
    }
    return {x, y};
}

// ---------------------------------------------------------------- HorizonEstimator

void HorizonEstimator::reset() {
    // pitch > 0 (looking down) puts the horizon above the image centre
    y_ = cfg_.cy - cfg_.fy * std::tan(deg2rad(cfg_.pitch_deg));
    accepted_ = rejected_streak_ = 0;
    lines_.clear();
}

double HorizonEstimator::sigma_px() const {
    if (!cfg_.auto_horizon) return 6.0;
    return locked() ? 4.0 : 15.0;
}

double HorizonEstimator::pitch_deg() const {
    double yn = cfg_.normalize({cfg_.cx, y_}).y;
    return rad2deg(std::atan(-yn));
}

void HorizonEstimator::update(const cv::Mat& frame) {
    lines_.clear();
    if (!cfg_.auto_horizon) return;

    // Work on a 640-px-wide copy for speed.
    const double s = 640.0 / frame.cols;
    cv::Mat small, gray, edges;
    cv::resize(frame, small, {}, s, s, cv::INTER_AREA);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, {5, 5}, 0);
    cv::Canny(gray, edges, 50, 150);
    const int H = small.rows, W = small.cols;

    // Search for lane markings between the current horizon and the bonnet.
    int top = std::clamp((int)(y_ * s + 0.03 * H), (int)(0.2 * H), (int)(0.65 * H));
    int bottom = std::clamp((int)(cfg_.bonnet_top_frac * H) - 2, top + 1, H);
    edges.rowRange(0, top).setTo(0);
    edges.rowRange(bottom, H).setTo(0);

    std::vector<cv::Vec4i> segs;
    cv::HoughLinesP(edges, segs, 1, kPi / 180, 25, 0.05 * W, 8);

    struct Seg { double x1, y1, m, len; };
    std::vector<Seg> left, right;
    for (auto& l : segs) {
        double dx = l[2] - l[0], dy = l[3] - l[1];
        if (std::fabs(dx) < 1) continue;
        double m = dy / dx, len = std::hypot(dx, dy), mid = (l[0] + l[2]) / 2.0;
        if (std::fabs(m) < 0.3 || std::fabs(m) > 3.0) continue;  // drop shadows/horizontals and poles
        if (m < 0 && mid < 0.6 * W) left.push_back({(double)l[0], (double)l[1], m, len});
        else if (m > 0 && mid > 0.4 * W) right.push_back({(double)l[0], (double)l[1], m, len});
        else continue;
        lines_.push_back(cv::Vec4f(l[0] / s, l[1] / s, l[2] / s, l[3] / s));
    }
    auto by_len = [](const Seg& a, const Seg& b) { return a.len > b.len; };
    std::sort(left.begin(), left.end(), by_len);
    std::sort(right.begin(), right.end(), by_len);
    if (left.size() > 12) left.resize(12);
    if (right.size() > 12) right.resize(12);

    // Vanishing point = intersections of left/right lane lines; weighted median of their rows.
    std::vector<std::pair<double, double>> ys;  // (row, weight)
    for (auto& a : left)
        for (auto& b : right) {
            double x = (a.m * a.x1 - b.m * b.x1 + b.y1 - a.y1) / (a.m - b.m);
            double y = a.y1 + a.m * (x - a.x1);
            if (x < 0.15 * W || x > 0.85 * W || y < 0.1 * H || y > 0.7 * H) continue;
            ys.push_back({y, a.len * b.len});
        }
    if (ys.empty()) return;
    std::sort(ys.begin(), ys.end());
    double total = 0, acc = 0, meas = ys[0].first;
    for (auto& p : ys) total += p.second;
    for (auto& p : ys) {
        acc += p.second;
        if (acc >= total / 2) { meas = p.first; break; }
    }
    meas /= s;

    const double gate = 0.06 * frame.rows;
    if (accepted_ == 0) {
        y_ = meas;
        accepted_ = 1;
    } else if (!locked() || std::fabs(meas - y_) < gate) {
        y_ += (locked() ? 0.15 : 0.3) * (meas - y_);
        ++accepted_;
        rejected_streak_ = 0;
    } else if (++rejected_streak_ > 30) {  // horizon really moved (e.g. slope): re-acquire
        y_ = meas;
        accepted_ = 1;
        rejected_streak_ = 0;
    }
}

// ---------------------------------------------------------------- DistanceEstimator

void DistanceEstimator::reset() {
    height_ = cfg_.camera_height_m;
    samples_.clear();
}

// Horizontal distance from the camera to where the pixel's ray meets a flat road.
double DistanceEstimator::ground_distance(double u, double v, double horizon_y) const {
    cv::Point2d p = cfg_.normalize({u, v});
    double theta = std::atan(-cfg_.normalize({cfg_.cx, horizon_y}).y);  // pitch, + = down
    // Rotate the camera ray into a level frame (y down, z forward).
    double dy = p.y * std::cos(theta) + std::sin(theta);
    double dz = -p.y * std::sin(theta) + std::cos(theta);
    if (dy <= 1e-4 || dz <= 0) return -1;  // at/above the horizon: never meets the road
    double t = height_ / dy;
    return std::hypot(t * p.x, t * dz);
}

DistanceEstimate DistanceEstimator::estimate(const Detection& d, const std::string& cls,
                                             const HorizonEstimator& hz) {
    DistanceEstimate e;
    const double u = d.box.x + d.box.width / 2.0, v = d.box.y + d.box.height;
    const double h_px = d.box.height;
    e.bonnet_cut = v >= cfg_.bonnet_top_frac * cfg_.image.height - 2;

    double var_g = -1, var_s = -1;
    if (!e.bonnet_cut) {
        double g = ground_distance(u, v, hz.y());
        if (g > 0 && g < cfg_.max_distance_m) {
            // Pixel error of the box bottom + horizon error -> distance error.
            double sv = std::sqrt(4.0 + std::pow(0.02 * h_px, 2) + std::pow(hz.sigma_px(), 2));
            double g_far = ground_distance(u, v - sv, hz.y());
            double sig = g_far > 0 ? std::fabs(g_far - g) : g;
            e.ground = g;
            var_g = sig * sig;
        }
    }

    auto it = cfg_.sizes.find(cls);
    if (it != cfg_.sizes.end() && h_px >= 6 && d.box.y > 1) {
        double s = cfg_.fy * it->second.height_m / h_px;
        if (s < cfg_.max_distance_m) {
            double rel = std::sqrt(std::pow(2.0 / h_px, 2) + std::pow(it->second.rel_sigma, 2));
            if (e.bonnet_cut) rel *= 2;  // part of the object hidden: box is too short
            e.size = s;
            var_s = std::pow(s * rel, 2);
        }
        // Self-calibrate camera height from objects whose size is well known.
        if (cfg_.auto_camera_height && e.ground > 0 && e.size > 0 && it->second.rel_sigma <= 0.06 &&
            h_px >= 20 && e.size < 40 && hz.locked()) {
            samples_.push_back(height_ * e.size / e.ground);
            if (samples_.size() > 300) samples_.pop_front();
            if (samples_.size() >= 15) {
                std::vector<double> tmp(samples_.begin(), samples_.end());
                std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
                height_ = std::clamp(tmp[tmp.size() / 2], 0.3, 3.0);
            }
        }
    }

    if (var_g > 0 && var_s > 0) {
        double wg = 1 / var_g, ws = 1 / var_s;
        e.fused = (wg * e.ground + ws * e.size) / (wg + ws);
        e.sigma = std::sqrt(1 / (wg + ws));
    } else if (var_g > 0) {
        e.fused = e.ground;
        e.sigma = std::sqrt(var_g);
    } else if (var_s > 0) {
        e.fused = e.size;
        e.sigma = std::sqrt(var_s);
    }
    return e;
}

// ---------------------------------------------------------------- Tracker

static float box_iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    return inter / (a.area() + b.area() - inter + 1e-6f);
}

void Tracker::update(const std::vector<Detection>& dets, std::vector<DistanceEstimate>& est) {
    std::vector<int> det_track(dets.size(), -1);
    std::vector<bool> used(tracks_.size(), false);

    // Greedy association by IoU (same class), best pairs first.
    struct Pair { float iou; int d, t; };
    std::vector<Pair> pairs;
    for (int i = 0; i < (int)dets.size(); ++i)
        for (int j = 0; j < (int)tracks_.size(); ++j)
            if (dets[i].cls == tracks_[j].cls) {
                float v = box_iou(dets[i].box, tracks_[j].box);
                if (v > 0.15f) pairs.push_back({v, i, j});
            }
    std::sort(pairs.begin(), pairs.end(), [](auto& a, auto& b) { return a.iou > b.iou; });
    for (auto& p : pairs)
        if (det_track[p.d] < 0 && !used[p.t]) { det_track[p.d] = p.t; used[p.t] = true; }

    for (int i = 0; i < (int)dets.size(); ++i) {
        if (det_track[i] < 0) {
            tracks_.push_back({next_id_++, dets[i].cls, dets[i].box});
            used.push_back(true);
            det_track[i] = (int)tracks_.size() - 1;
        }
        Track& t = tracks_[det_track[i]];
        t.box = dets[i].box;
        t.missed = 0;
        DistanceEstimate& e = est[i];
        e.track_id = t.id;
        if (e.fused > 0) {
            // 1-D Kalman filter on distance (random-walk model).
            double r = e.sigma * e.sigma;
            if (t.z < 0) {
                t.z = e.fused;
                t.p = r;
            } else {
                t.p += std::pow(0.08 * t.z, 2);
                double k = t.p / (t.p + r);
                t.z += k * (e.fused - t.z);
                t.p *= (1 - k);
            }
        }
        e.smoothed = t.z;
    }

    for (size_t j = 0; j < tracks_.size(); ++j)
        if (!used[j]) ++tracks_[j].missed;
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const Track& t) { return t.missed > 10; }),
                  tracks_.end());
}
