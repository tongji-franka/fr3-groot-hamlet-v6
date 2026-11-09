#pragma once
#include "../fr3_replay/trajectory.hpp"
#include "act_limits.hpp"
#include <ruckig/ruckig.hpp>

namespace fr3_act {
using namespace fr3_replay;
struct Sample { Q q{}, v{}, a{}; };
struct MotionLimits {
  double speed_scale;
  Q velocity=velocity_limits, acceleration=acceleration_limits, jerk=jerk_limits;
  explicit MotionLimits(double scale=1.0) : speed_scale(scale) {
    require(std::isfinite(scale) && scale>=.1 && scale<=1,
            "speed-scale must be finite and within [0.1, 1]");
    constexpr Q hardware_velocity{2.62,2.62,2.62,2.62,5.26,4.18,5.26};
    for(size_t j=0;j<7;++j) {
      require(std::isfinite(velocity[j]) && velocity[j]>0 && velocity[j]<hardware_velocity[j]
              && std::isfinite(acceleration[j]) && acceleration[j]>0 && acceleration[j]<10
              && std::isfinite(jerk[j]) && jerk[j]>0 && jerk[j]<5000,
              "Invalid ACT profile; limits must remain below FR3 nominal maxima");
      velocity[j]*=scale;
      acceleration[j]*=scale*scale;
      jerk[j]*=scale*scale*scale;
    }
  }
};
class Motion {
  ruckig::Ruckig<7> planner{0.001};
  ruckig::InputParameter<7> limits_input;
  ruckig::Trajectory<7> trajectory, brake;
  Q target_position{}, target_velocity{};
  double elapsed=0;
  bool active=false;

  static bool inside(ruckig::Trajectory<7>& path) {
    auto profiles=path.get_profiles();
    for (size_t j=0; j<7; ++j) {
      auto& profile=profiles[0][j];
      // Ruckig 0.9.2 leaves pf=0 in velocity mode; extrema must use the integrated endpoint.
      profile.pf=profile.p.back();
      const auto extrema=profile.get_position_extrema();
      if (!std::isfinite(extrema.min) || !std::isfinite(extrema.max)
          || extrema.min<lower[j] || extrema.max>upper[j]) return false;
    }
    return true;
  }
  ruckig::InputParameter<7> from(const Sample& start) const {
    auto input=limits_input;
    input.current_position=start.q;
    input.current_velocity=start.v;
    input.current_acceleration=start.a;
    return input;
  }
  bool stopping_path(const Sample& start, ruckig::Trajectory<7>& path) {
    auto input=from(start);
    input.control_interface=ruckig::ControlInterface::Velocity;
    input.target_velocity.fill(0);
    input.target_acceleration.fill(0);
    return planner.calculate(input,path)>=0 && inside(path);
  }

 public:
  const MotionLimits limits;
  Sample state;
  bool stopping=false, finished=false, refused=false;
  explicit Motion(const Q& q, double speed_scale=1.0) : limits(speed_scale), state{q,{},{}} {
    check_q(q);
    limits_input.max_velocity=limits.velocity;
    Q minimum_velocity{};
    for(size_t j=0;j<7;++j) minimum_velocity[j]=-limits.velocity[j];
    limits_input.min_velocity=minimum_velocity;
    limits_input.max_acceleration=limits.acceleration;
    limits_input.max_jerk=limits.jerk;
    limits_input.synchronization=ruckig::Synchronization::None;
    require(stopping_path(state,brake),"Initial pose has no bounded stop");
  }
  void set_velocity_bounds(const Q& minimum, const Q& maximum) {
    Q lower_velocity{};
    for(size_t j=0;j<7;++j) {
      require(std::isfinite(minimum[j]) && minimum[j]<0
              && std::isfinite(maximum[j]) && maximum[j]>0,
              "No bidirectional SDK joint velocity envelope");
      // Reserve 5% for the SDK filter/echo delay as the position-dependent bound moves.
      limits_input.max_velocity[j]=std::min(limits.velocity[j],.95*maximum[j]);
      lower_velocity[j]=std::max(-limits.velocity[j],.95*minimum[j]);
    }
    limits_input.min_velocity=lower_velocity;
  }
  bool retarget(const Q& target, const Q& velocity={}) {
    if(stopping) return false;
    for(size_t j=0;j<7;++j)
      if(!std::isfinite(target[j]) || target[j]<lower[j] || target[j]>upper[j]
          || !std::isfinite(velocity[j]) || std::abs(velocity[j])>limits.velocity[j]) return false;
    target_position=target;
    target_velocity=velocity;
    elapsed=0; active=true;
    return true;
  }
  void stop() {
    if(stopping) return;
    // brake was checked at the last emitted state, including its full position extrema.
    trajectory=brake;
    elapsed=0; active=true; stopping=true;
  }
  Sample tick(double dt) {
    require(std::isfinite(dt) && dt>0 && dt<=0.01,"Invalid motion period");
    if(!active) return state;
    Sample next;
    if(stopping) trajectory.at_time(elapsed+dt,next.q,next.v,next.a);
    else {
      auto input=from(state);
      input.control_interface=ruckig::ControlInterface::Velocity;
      // Coordinate all axes instead of independently accelerating toward a new chunk.
      input.synchronization=ruckig::Synchronization::Time;
      input.target_acceleration.fill(0);
      double common_scale=1.;
      for(size_t j=0;j<7;++j) {
        // ACT action is q(t+1); interpolate from the current tick using the chunk slope.
        const double reference=target_position[j]+target_velocity[j]*(elapsed-1./(30*limits.speed_scale));
        input.target_velocity[j]=target_velocity[j]+12.*limits.speed_scale*(reference-state.q[j]);
        // Velocity-control Ruckig does not clamp target_velocity to max_velocity.
        const double bound=input.target_velocity[j]>=0 ? limits_input.max_velocity[j]
                                                       : -(*limits_input.min_velocity)[j];
        if(std::abs(input.target_velocity[j])>bound)
          common_scale=std::min(common_scale,bound/std::abs(input.target_velocity[j]));
      }
      for(double& velocity:input.target_velocity) velocity*=common_scale;
      if(planner.calculate(input,trajectory)<0) { refused=true; stop(); return tick(dt); }
      trajectory.at_time(dt,next.q,next.v,next.a);
    }
    bool valid=true;
    for(size_t j=0;j<7;++j)
      valid=valid && std::isfinite(next.q[j]) && next.q[j]>=lower[j] && next.q[j]<=upper[j]
        && std::isfinite(next.v[j]) && std::abs(next.v[j])<=limits.velocity[j]+1e-7
        && std::isfinite(next.a[j]) && std::abs(next.a[j])<=limits.acceleration[j]+1e-7
        && std::abs(next.a[j]-state.a[j])<=limits.jerk[j]*dt+1e-7
        && std::abs(next.v[j]+std::copysign(next.a[j]*next.a[j]/(2*limits.jerk[j]),next.a[j]))<=limits.velocity[j]+1e-7;
    ruckig::Trajectory<7> next_brake;
    if(!stopping && (!valid || !stopping_path(next,next_brake))) {
      refused=true; stop();
      return tick(dt);
    }
    require(valid,"Braking trajectory violates ACT limits");
    elapsed+=dt;
    state=next;
    if(!stopping) brake=next_brake;
    else finished=elapsed>=trajectory.get_duration();
    return state;
  }
};
}  // namespace fr3_act
