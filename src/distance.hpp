// Monocular camera-to-object distance estimation.
//
// Two independent estimates per detection, fused by inverse variance:
//   1. Ground plane: the box's bottom-centre lies on a flat road; with camera height and the
//      horizon (pitch) the ray hits the road at a unique distance.
//   2. Known size:   distance = focal_px * real_height_m / box_height_px.
// The horizon is re-estimated every frame from the lane-line vanishing point (tracks pitch
// changes as the car bounces), and the camera height can self-calibrate from objects of
// well-known size. A small IoU tracker smooths each object's distance over time.
#pragma once

#include <opencv2/core.hpp>

#include <deque>
#include <map>
#include <string>
#include <vector>

struct Detection {
    cv::Rect2f box;
    float score;
    int cls;
};

struct ObjectSize {
    double height_m;   // real-world height
    double rel_sigma;  // relative uncertainty of that height (0.05 = +-5%)
};

struct CameraConfig {
    // Intrinsics: if fx == 0 they are derived from hfov_deg (approximation - calibrate!).
    double hfov_deg = 110.0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::vector<double> dist;  // OpenCV k1 k2 p1 p2 [k3]; empty = no distortion
    int calib_width = 0, calib_height = 0;  // resolution fx..cy were calibrated at (0 = same as video)

    double camera_height_m = 1.2;  // lens above road
    bool auto_camera_height = true;
    double pitch_deg = 0.0;  // initial pitch (positive = looking down)
    bool auto_horizon = true;
    double bonnet_top_frac = 0.72;  // rows below this are car bonnet
    double max_distance_m = 150.0;
    std::map<std::string, ObjectSize> sizes;

    cv::Size image;

    static CameraConfig load(const std::string& path, bool* found = nullptr);
    void finalize(cv::Size image_size);  // fill fx/fy/cx/cy for this resolution
    cv::Point2d normalize(cv::Point2d px) const;  // pixel -> undistorted normalised coords
};

class HorizonEstimator {
public:
    explicit HorizonEstimator(const CameraConfig& cfg) : cfg_(cfg) { reset(); }
    void reset();
    void update(const cv::Mat& frame);
    double y() const { return y_; }  // horizon row, pixels
    bool locked() const { return accepted_ >= 5; }
    double sigma_px() const;
    double pitch_deg() const;
    const std::vector<cv::Vec4f>& lane_lines() const { return lines_; }

private:
    const CameraConfig& cfg_;
    double y_ = 0;
    int accepted_ = 0, rejected_streak_ = 0;
    std::vector<cv::Vec4f> lines_;
};

struct DistanceEstimate {
    double ground = -1;    // ground-plane estimate, m (-1 = n/a)
    double size = -1;      // known-size estimate, m
    double fused = -1;     // inverse-variance fusion of the above
    double sigma = -1;     // 1-sigma uncertainty of fused, m
    double smoothed = -1;  // after temporal tracking
    bool bonnet_cut = false;  // bottom hidden by the bonnet -> size-only, lower confidence
    int track_id = -1;
};

class DistanceEstimator {
public:
    explicit DistanceEstimator(const CameraConfig& cfg) : cfg_(cfg) { reset(); }
    void reset();
    DistanceEstimate estimate(const Detection& d, const std::string& cls, const HorizonEstimator& hz);
    double camera_height() const { return height_; }
    int height_samples() const { return (int)samples_.size(); }

private:
    double ground_distance(double u, double v, double horizon_y) const;
    const CameraConfig& cfg_;
    double height_ = 1.2;
    std::deque<double> samples_;
};

class Tracker {
public:
    void reset() { tracks_.clear(); next_id_ = 1; }
    // Associates detections with tracks and fills est[i].track_id / est[i].smoothed.
    void update(const std::vector<Detection>& dets, std::vector<DistanceEstimate>& est);

private:
    struct Track {
        int id, cls;
        cv::Rect2f box;
        double z = -1, p = 0;
        int missed = 0;
    };
    std::vector<Track> tracks_;
    int next_id_ = 1;
};
