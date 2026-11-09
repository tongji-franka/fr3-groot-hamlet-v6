#pragma once
#include "act_limits.hpp"
#include "../fr3_replay/trajectory.hpp"

namespace fr3_act {
template<size_t N>
void check_collision_pair(const std::array<double,N>& lower, const std::array<double,N>& upper) {
  for(size_t i=0;i<N;++i)
    fr3_replay::require(std::isfinite(lower[i]) && std::isfinite(upper[i])
                        && lower[i]>=0 && upper[i]>0 && lower[i]<=upper[i],
                        "Invalid collision thresholds: require finite 0 <= lower <= upper, upper > 0");
}
inline void validate_collision_thresholds() {
  check_collision_pair(collision_lower_torque_acceleration,collision_upper_torque_acceleration);
  check_collision_pair(collision_lower_torque_nominal,collision_upper_torque_nominal);
  check_collision_pair(collision_lower_force_acceleration,collision_upper_force_acceleration);
  check_collision_pair(collision_lower_force_nominal,collision_upper_force_nominal);
}
template<class Robot>
void configure_collision(Robot& robot) {
  validate_collision_thresholds();
  robot.setCollisionBehavior(
    collision_lower_torque_acceleration,collision_upper_torque_acceleration,
    collision_lower_torque_nominal,collision_upper_torque_nominal,
    collision_lower_force_acceleration,collision_upper_force_acceleration,
    collision_lower_force_nominal,collision_upper_force_nominal);
}
}
