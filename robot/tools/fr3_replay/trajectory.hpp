#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fr3_replay {
using Q = std::array<double, 7>;
constexpr Q lower{-2.8007, -1.7361, -2.8007, -2.9770, -2.7763, 0.5398, -2.9508};
constexpr Q upper{2.8007, 1.7361, 2.8007, -0.2169, 2.7763, 4.5216, 2.9508};
constexpr std::array<double, 3> limits{0.25, 1.0, 20.0};
inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
inline void check_q(const Q& q) {
  for (size_t j = 0; j < 7; ++j)
    require(std::isfinite(q[j]) && q[j] >= lower[j] && q[j] <= upper[j],
            "Position outside FR3 ranges with 0.1 rad margin");
}
inline double distance(const Q& a, const Q& b) {
  double result = 0;
  for (size_t j = 0; j < 7; ++j) {
    require(std::isfinite(a[j]) && std::isfinite(b[j]), "Non-finite robot state");
    result = std::max(result, std::abs(a[j] - b[j]));
  }
  return result;
}
struct StartState { Q q, q_d, dq, dq_d, ddq_d; bool idle; };
inline double check_start(const StartState& s, const Q& first_q, bool move_to_start) {
  require(s.idle, "Execution requires Idle with active FCI, not Guiding/User stopped; use Desk's normal execution workflow");
  check_q(s.q);
  check_q(s.q_d);
  for (size_t j = 0; j < 7; ++j)
    require(std::abs(s.dq[j]) < 0.02 && std::abs(s.dq_d[j]) < 0.02 && std::abs(s.ddq_d[j]) < 0.1,
            "Robot must be stationary before replay");
  require(distance(s.q, s.q_d) <= 0.05, "Measured and commanded initial poses disagree");
  double error = distance(s.q, first_q);
  require(error <= 0.05 || move_to_start,
          "Start pose mismatch >0.05 rad; manually align or explicitly add --move-to-start after clearing the path");
  return error;
}
struct Point { double t; Q q; };
struct Segment { double t, duration; std::array<Q, 6> b; };
class Trajectory {
 public:
  std::vector<Point> points;
  std::vector<Segment> segments;
  std::array<double, 3> bounds{};
  explicit Trajectory(std::vector<Point> input) : points(std::move(input)) {
    require(points.size() >= 2 && points.size() <= 100000, "Invalid waypoint count");
    require(std::abs(points.front().t) < 1e-9, "Trajectory must start at zero");
    require(points.back().t > 0 && points.back().t <= 3600, "Invalid trajectory duration");
    std::vector<Q> slopes(points.size() - 1), velocity(points.size(), Q{});
    for (size_t i = 0; i < points.size(); ++i) {
      check_q(points[i].q);
      require(std::isfinite(points[i].t), "Non-finite source time");
      if (i) {
        double dt = points[i].t - points[i - 1].t;
        require(dt > 0, "Non-increasing source time");
        for (size_t j = 0; j < 7; ++j) slopes[i - 1][j] = (points[i].q[j] - points[i - 1].q[j]) / dt;
      }
    }
    for (size_t i = 1; i + 1 < points.size(); ++i)
      for (size_t j = 0; j < 7; ++j) {
        double a = slopes[i - 1][j], b = slopes[i][j];
        if (a * b > 0) velocity[i][j] = std::copysign(std::min(std::abs(a), std::abs(b)), a);
      }
    for (size_t i = 0; i + 1 < points.size(); ++i) {
      Segment s{points[i].t, points[i + 1].t - points[i].t, {}};
      for (size_t j = 0; j < 7; ++j) {
        double a = points[i].q[j], b = points[i + 1].q[j];
        double left = velocity[i][j] * s.duration / 5, right = velocity[i + 1][j] * s.duration / 5;
        s.b[0][j] = a; s.b[1][j] = a + left; s.b[2][j] = a + 2 * left;
        s.b[3][j] = b - 2 * right; s.b[4][j] = b - right; s.b[5][j] = b;
      }
      auto polygon = s.b;
      for (size_t order = 0; order < 3; ++order) {
        size_t degree = 5 - order;
        for (size_t k = 0; k < degree; ++k)
          for (size_t j = 0; j < 7; ++j) {
            polygon[k][j] = degree * (polygon[k + 1][j] - polygon[k][j]) / s.duration;
            require(std::isfinite(polygon[k][j]), "Non-finite derivative");
            bounds[order] = std::max(bounds[order], std::abs(polygon[k][j]));
          }
      }
      segments.push_back(s);
    }
  }
  void validate_limits() const {
    for (size_t i = 0; i < 3; ++i)
      require(bounds[i] <= limits[i] * (1 + 1e-7), "Trajectory exceeds replay velocity/acceleration/jerk caps");
  }
  Q at(double time) const {
    if (time <= 0) return points.front().q;
    if (time >= points.back().t) return points.back().q;
    auto it = std::upper_bound(segments.begin(), segments.end(), time,
                              [](double t, const Segment& s) { return t < s.t; });
    const auto& s = *std::prev(it);
    double u = std::clamp((time - s.t) / s.duration, 0., 1.);
    auto work = s.b;
    for (size_t level = 5; level > 0; --level)
      for (size_t k = 0; k < level; ++k)
        for (size_t j = 0; j < 7; ++j) work[k][j] = (1 - u) * work[k][j] + u * work[k + 1][j];
    return work[0];
  }
};
inline Trajectory load(const std::string& path) {
  std::ifstream input(path);
  require(input.good(), "Cannot open replay plan");
  std::string magic;
  std::getline(input, magic);
  require(magic == "FR3_REPLAY_V1", "Invalid replay plan header");
  size_t count{};
  require(static_cast<bool>(input >> count) && count >= 2 && count <= 100000, "Invalid waypoint count");
  std::vector<Point> points(count);
  for (auto& p : points) {
    require(static_cast<bool>(input >> p.t), "Missing source time");
    for (double& value : p.q) require(static_cast<bool>(input >> value), "Missing joint value");
  }
  std::string trailing;
  require(!(input >> trailing), "Trailing data in replay plan");
  Trajectory result(std::move(points));
  result.validate_limits();
  return result;
}
inline Trajectory approach(const Q& current, const Q& target) {
  Trajectory first({{0., current}, {2., target}});
  double factor = std::max({1., first.bounds[0] / limits[0],
      std::sqrt(first.bounds[1] / limits[1]), std::cbrt(first.bounds[2] / limits[2])}) * 1.01;
  Trajectory result({{0., current}, {2 * factor, target}});
  result.validate_limits();
  return result;
}
// Unit-duration cubic time deceleration: rate 1 -> 0, zero endpoint rate derivatives.
inline double stop_integral(double t) {
  t = std::clamp(t, 0., 1.);
  return t - t * t * t + 0.5 * t * t * t * t;
}
}  // namespace fr3_replay
