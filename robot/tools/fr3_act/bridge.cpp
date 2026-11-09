#include "motion.hpp"
#include "collision.hpp"
#include "act_build_info.hpp"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>
#ifdef FR3_ACT_ROBOT
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#endif

namespace {
using namespace fr3_act;
using Clock = std::chrono::steady_clock;
double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }
// Version 6 appends native receive/accept/output times, robot time and publication time.
using Command = std::array<double, 17>;
using TimedCommand = std::array<double, 18>;
using State = std::array<double, 41>;
template<size_t N> struct Mailbox {
  static_assert(std::atomic<double>::is_always_lock_free);
  std::atomic<unsigned> revision{0};
  std::array<std::atomic<double>, N> data{};
  void write(const std::array<double, N>& values) {
    revision.fetch_add(1);
    for (size_t i=0; i<N; ++i) data[i].store(values[i]);
    revision.fetch_add(1);
  }
  bool read(std::array<double, N>& values) const {
    unsigned before = revision.load();
    if (before & 1) return false;
    for (size_t i=0; i<N; ++i) values[i] = data[i].load();
    return before == revision.load();
  }
};
struct Link {
  const MotionLimits limits;
  int fd = -1;
  Mailbox<18> command;
  Mailbox<41> state;
  std::atomic<bool> stop{false}, done{false};
  std::array<char,2048> failure{};
  std::atomic<size_t> failure_size{0};
  std::thread worker;
  explicit Link(int port, double speed_scale=1.0) : limits(speed_scale) {
    fd = socket(AF_INET, SOCK_DGRAM, 0);
    require(fd >= 0, "Cannot create UDP socket");
    sockaddr_in address{};
    address.sin_family=AF_INET; address.sin_port=htons(port); address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
      close(fd); throw std::runtime_error("Cannot bind loopback UDP port");
    }
    worker=std::thread([this] { io(); });
  }
  ~Link() { done=true; if(worker.joinable()) worker.join(); if(fd>=0) close(fd); }
  void fail(const char* reason) {
    const std::string message=std::string("FR3_ACT_ERROR: ")+reason;
    const size_t size=std::min(message.size(),failure.size());
    std::copy_n(message.begin(),size,failure.begin());
    failure_size.store(size);
    stop=true;
    // Outside the real-time callback: allow several nonblocking deliveries before closing UDP.
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }
  void io() {
    sockaddr_in peer{};
    bool connected=false;
    double latest_seq=0;
    while (!done) {
      // Bounded drain: a sender cannot starve the watchdog/state publisher.
      for (int i=0; i<32; ++i) {
        Command packet{};
        sockaddr_in source{};
        socklen_t size=sizeof(source);
        auto n=recvfrom(fd, packet.data(), sizeof(packet), MSG_DONTWAIT|MSG_TRUNC,
                        reinterpret_cast<sockaddr*>(&source), &size);
        if (n<0) break;
        if (!connected) {
          if (n!=sizeof(packet) || packet[0]!=32000 || packet[1]!=protocol_version) continue;
          peer=source; connected=true;
        }
        if (source.sin_port!=peer.sin_port || source.sin_addr.s_addr!=peer.sin_addr.s_addr) continue;
        if (n!=sizeof(packet)) { stop=true; continue; }
        if (packet[0]==32000) continue;
        if (packet[0]==32003) { stop=true; continue; }
        bool valid=packet[0]==32001 && packet[1]>latest_seq && packet[1]<9007199254740992.
                   && std::floor(packet[1])==packet[1];
        double stamp=now();
        valid=valid && packet[2]>stamp && packet[2]<=stamp+watchdog_seconds;
        for (double x : packet) valid=valid && std::isfinite(x);
        Q q{};
        std::copy(packet.begin()+3, packet.begin()+10, q.begin());
        for (size_t j=0; j<7; ++j)
          valid=valid && q[j]>=lower[j] && q[j]<=upper[j] && std::abs(packet[j+10])<=limits.velocity[j];
        if (!valid) { stop=true; continue; }
        latest_seq=packet[1];
        TimedCommand timed{};
        std::copy(packet.begin(),packet.end(),timed.begin());
        timed[17]=stamp;
        command.write(timed);
      }
      if (connected) {
        const size_t failed=failure_size.load();
        if(failed) sendto(fd,failure.data(),failed,MSG_DONTWAIT,reinterpret_cast<sockaddr*>(&peer),sizeof(peer));
        else {
          State packet{};
          if (state.read(packet) && packet[0]==32002)
            sendto(fd, packet.data(), sizeof(packet), MSG_DONTWAIT,
                   reinterpret_cast<sockaddr*>(&peer), sizeof(peer));
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
};
struct Controller {
  Link& link;
  Motion motion;
  double sequence=0, deadline=0, settle=0;
  double received_stamp=0, accepted_stamp=0, output_stamp=0;
  double max_start_distance, max_target_distance;
  Q accepted_target{};
  bool started=false, finished=false, refused=false;
  explicit Controller(Link& value, const Q& q, double start_distance, double target_distance=1.0)
      : link(value), motion(q,value.limits.speed_scale), max_start_distance(start_distance), max_target_distance(target_distance) {}
  Q tick(const Q& measured, const Q& commanded, double dt) {
    require(std::isfinite(dt) && dt>0 && dt<=0.01, "Control callback timing gap exceeds 10 ms");
    require(distance(measured, commanded)<=0.1, "Joint tracking error exceeds 0.1 rad");
    if (link.stop || interrupted || (started && now()>deadline)) motion.stop();
    const double previous_sequence=sequence;
    TimedCommand packet{};
    if (!motion.stopping && link.command.read(packet) && packet[1]>sequence) {
      Q target{}, velocity{};
      std::copy(packet.begin()+3, packet.begin()+10, target.begin());
      std::copy(packet.begin()+10, packet.begin()+17, velocity.begin());
      bool valid=packet[2]>now();
      for(size_t j=0;j<7;++j)
        valid=valid && std::abs(measured[j]-target[j])<=(started ? (max_target_distance>0 ? max_target_distance : 1.25*std::max(.10,.05+.075*velocity_limits[j])) : max_start_distance);
      if (!valid) {
        motion.stop(); refused=true;
      } else {
        if (!motion.retarget(target,velocity)) { motion.stop(); refused=true; }
        else {
          sequence=packet[1]; deadline=packet[2]; started=true; accepted_target=target;
          received_stamp=packet[17]; accepted_stamp=now();
        }
      }
    }
    const auto output=motion.tick(dt);
    if(sequence!=previous_sequence) output_stamp=now();
    refused=refused || motion.refused;
    if(motion.finished) settle+=dt;
    finished=settle>=0.05;
    return output.q;
  }
};
void publish(Link& link, const Q& q, const Q& qd, bool real, const Controller* controller=nullptr,
             bool read_only=false, double received_stamp=0, double robot_time=0) {
  State state{};
  const double published=now();
  state[0]=32002; state[1]=real ? 1 : 0; state[2]=received_stamp>0 ? received_stamp : published; state[3]=profile_id;
  std::copy(q.begin(),q.end(),state.begin()+4);
  std::copy(qd.begin(),qd.end(),state.begin()+11);
  state[18]=link.limits.speed_scale;
  state[35]=protocol_version;
  state[39]=robot_time;
  state[40]=published;
  state[34]=read_only ? 4 : 0;
  if(controller) {
    state[19]=controller->sequence;
    state[36]=controller->received_stamp;
    state[37]=controller->accepted_stamp;
    state[38]=controller->output_stamp;
    std::copy(controller->accepted_target.begin(),controller->accepted_target.end(),state.begin()+20);
    std::copy(controller->motion.state.q.begin(),controller->motion.state.q.end(),state.begin()+27);
    state[34]=controller->finished ? 3 : controller->motion.stopping ? 2 : controller->started ? 1 : 0;
  } else {
    std::copy(qd.begin(),qd.end(),state.begin()+27);
  }
  link.state.write(state);
}
}

int main(int argc, char** argv) {
  // This path must never create a socket or contact the robot.
  if (argc==2 && std::string(argv[1])=="--build-info") {
    std::cout << "{\"protocol\":" << protocol_version
              << ",\"watchdog_seconds\":" << watchdog_seconds
              << ",\"source_sha256\":\"" << build_source_sha256
              << "\",\"profile_sha256\":\"" << build_profile_sha256 << "\"}\n";
    return 0;
  }
  std::unique_ptr<Link> link_owner;
  try {
    int port=8768;
    double max_start_distance=1.0;
    double max_target_distance=1.0;
    double speed_scale=1.0;
    double observe_duration=60.0;
    bool execute=false, observe_only=false;
    std::string ip;
    for (int i=1; i<argc; ++i) {
      std::string arg=argv[i];
      if (arg=="--port" && i+1<argc) port=std::stoi(argv[++i]);
      else if (arg=="--max-start-distance" && i+1<argc) max_start_distance=std::stod(argv[++i]);
      else if (arg=="--speed-scale" && i+1<argc) speed_scale=std::stod(argv[++i]);
      else if (arg=="--observe-duration" && i+1<argc) observe_duration=std::stod(argv[++i]);
      else if (arg=="--max-target-distance" && i+1<argc) max_target_distance=std::stod(argv[++i]);
      else if (arg=="--execute") execute=true;
      else if (arg=="--observe-only") observe_only=true;
      else if (arg=="--robot-ip" && i+1<argc) ip=argv[++i];
      else throw std::runtime_error("Usage: fr3_act_sim [--port N] [--max-start-distance RAD] [--speed-scale S] [--max-target-distance RAD] OR fr3_act_bridge --robot-ip IP (--execute|--observe-only) [--port N] [--max-start-distance RAD] [--speed-scale S] [--max-target-distance RAD] [--observe-duration SECONDS (0=continuous, observe-only)]");
    }
    require(port>=1024 && port<=65535,"Invalid port");
    require(std::isfinite(max_start_distance) && max_start_distance>0 && max_start_distance<=1,
            "max-start-distance must be finite and within (0, 1] rad");
    require(std::isfinite(observe_duration) && observe_duration>=0,
            "observe-duration must be finite and nonnegative (0 means continuous)");
    require(observe_only || observe_duration==60.0,"observe-duration requires --observe-only");
    require(std::isfinite(max_target_distance) && max_target_distance>=0 && max_target_distance<=1,
            "max-target-distance must be finite and within [0, 1] rad; 0 uses profile limits");
    validate_collision_thresholds();
    const unsigned short endian=1;
    require(*reinterpret_cast<const unsigned char*>(&endian)==1,"Requires little-endian host");
#ifdef FR3_ACT_ROBOT
    require((execute != observe_only) && !ip.empty(),"Real bridge requires --robot-ip IP and exactly one of --execute / --observe-only");
    in_addr parsed{};
    require(inet_pton(AF_INET,ip.c_str(),&parsed)==1,"Robot IP must be IPv4");
#else
    require(!execute && !observe_only && ip.empty(),"Simulation does not accept --execute, --observe-only or --robot-ip");
#endif
    std::signal(SIGINT,interrupt); std::signal(SIGTERM,interrupt); std::signal(SIGHUP,interrupt);
    link_owner=std::make_unique<Link>(port,speed_scale);
    Link& link=*link_owner;
    std::cout << "ACT bridge on 127.0.0.1:" << port << "; dataset profile " << profile_id
              << ", Ruckig bounded braking, watchdog " << watchdog_seconds*1000 << " ms\n";
    std::cout << "Startup target-distance limit: " << max_start_distance << " rad\n";
    std::cout << "Running target-distance limit: " << max_target_distance << " rad (0 uses per-joint profile limits)\n";
    std::cout << "Speed scale: " << speed_scale << "; v*s, a*s^2, jerk*s^3; 1 kHz control\n";
    for(size_t j=0;j<7;++j)
      std::cout << "J" << j+1 << ": v=" << link.limits.velocity[j] << ", a=" << link.limits.acceleration[j]
                << ", jerk=" << link.limits.jerk[j] << '\n';
    std::cout << std::flush;
#ifdef FR3_ACT_ROBOT
    franka::Robot robot(ip,franka::RealtimeConfig::kEnforce);
    auto initial=robot.readOnce();
    double initial_received=now();
    if(observe_only) {
      std::cout << "READ ONLY: robot.control is disabled; observe-duration=" << observe_duration << " s (0 means continuous).\n"
                << "Configured m_ee_kg=" << initial.m_ee << ", m_load_kg=" << initial.m_load
                << ", m_total_kg=" << initial.m_total << "; load_com_flange_m=["
                << initial.F_x_Cload[0] << ',' << initial.F_x_Cload[1] << ',' << initial.F_x_Cload[2] << "]\n" << std::flush;
      const double begin=now();
      while(!link.stop && !interrupted && (observe_duration==0 || now()-begin<observe_duration)) {
        publish(link,initial.q,initial.q_d,true,nullptr,true,initial_received,initial.time.toSec());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        initial=robot.readOnce();
        initial_received=now();
      }
      std::cout << "Read-only state stream stopped\n";
      return 0;
    }
    check_start({initial.q,initial.q_d,initial.dq,initial.dq_d,initial.ddq_d,
                 initial.robot_mode==franka::RobotMode::kIdle},initial.q,false);
    // Configuration is applied only after the client has validated its model and sent a fresh goal.
    auto print_thresholds=[](const char* name,const auto& values) {
      std::cout << name << "=[";
      for(size_t i=0;i<values.size();++i) std::cout << (i ? "," : "") << values[i];
      std::cout << "]\n";
    };
    std::cout << "Pending collision profile (not yet applied; lower=contact, upper=reflex stop):\n";
    print_thresholds("lower_force_acceleration_N_Nm",collision_lower_force_acceleration);
    print_thresholds("upper_force_acceleration_N_Nm",collision_upper_force_acceleration);
    print_thresholds("lower_force_nominal_N_Nm",collision_lower_force_nominal);
    print_thresholds("upper_force_nominal_N_Nm",collision_upper_force_nominal);
    print_thresholds("lower_joint_torque_acceleration_Nm",collision_lower_torque_acceleration);
    print_thresholds("upper_joint_torque_acceleration_Nm",collision_upper_torque_acceleration);
    print_thresholds("lower_joint_torque_nominal_Nm",collision_lower_torque_nominal);
    print_thresholds("upper_joint_torque_nominal_Nm",collision_upper_torque_nominal);
    std::cout << std::flush;
    publish(link,initial.q,initial.q_d,true,nullptr,false,initial_received,initial.time.toSec());
    // Camera/model warmup happens before the first action and before entering robot.control.
    const double waiting=now();
    TimedCommand first{};
    while (!link.stop && !interrupted && now()-waiting<60) {
      if (link.command.read(first) && first[1]>0) break;
      initial=robot.readOnce();
      initial_received=now();
      publish(link,initial.q,initial.q_d,true,nullptr,false,initial_received,initial.time.toSec());
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (link.stop || interrupted) return 0;
    require(first[1]>0 && first[2]>now(),"No fresh first command within 60 seconds");
    initial=robot.readOnce();
    Q first_target{}; std::copy(first.begin()+3,first.begin()+10,first_target.begin());
    check_start({initial.q,initial.q_d,initial.dq,initial.dq_d,initial.ddq_d,
                 initial.robot_mode==franka::RobotMode::kIdle},initial.q,false);
    check_q(first_target);
    require(distance(initial.q,first_target)<=max_start_distance,"First ACT target exceeds max-start-distance");
    configure_collision(robot);
    std::cout << "Collision profile applied after first-goal admission\n" << std::flush;
    initial=robot.readOnce();
    check_start({initial.q,initial.q_d,initial.dq,initial.dq_d,initial.ddq_d,
                 initial.robot_mode==franka::RobotMode::kIdle},initial.q,false);
    require(!link.stop && !interrupted && first[2]>now(),"First goal expired during robot configuration");
    // The SDK can reset q_d when entering control; seed from its first callback.
    std::optional<Controller> controller;
    robot.control([&](const franka::RobotState& state, franka::Duration period) -> franka::JointPositions {
      const double received=now();
      if (!controller) {
        require(distance(state.q_d,initial.q_d)<0.005,"Robot moved after preflight");
        require(distance(state.q,state.q_d)<=0.05,"Initial measured and commanded poses disagree");
        require(distance(state.dq_d,Q{})<=1e-6 && distance(state.ddq_d,Q{})<=1e-6,
                "Control must start with stationary desired joints");
        controller.emplace(link,state.q_d,max_start_distance,max_target_distance);
        publish(link,state.q,state.q_d,true,&*controller,false,received,state.time.toSec());
        return franka::JointPositions(state.q_d);  // t=0: no synthetic 1 ms advance.
      }
      controller->motion.set_velocity_bounds(robot.getLowerJointVelocityLimits(state.q_d),
                                              robot.getUpperJointVelocityLimits(state.q_d));
      Q target=controller->tick(state.q,state.q_d,period.toSec());
      publish(link,state.q,state.q_d,true,&*controller,false,received,state.time.toSec());
      franka::JointPositions command(target);
      return controller->finished ? franka::MotionFinished(command) : command;
    },franka::ControllerMode::kJointImpedance,true,100.0);
    require(controller && !controller->refused,"ACT target refused; session stopped");
#else
    Q q{0.,0.,0.,-1.5,0.,1.5,0.};
    Controller controller(link,q,max_start_distance,max_target_distance);
    auto next=Clock::now();
    double simulated_time=0;
    while (!controller.finished) {
      const auto started=Clock::now();
      q=controller.tick(q,q,0.001);
      simulated_time+=.001;
      publish(link,q,q,false,&controller,false,std::chrono::duration<double>(started.time_since_epoch()).count(),simulated_time);
      // Simulation never bursts through overdue slots; it is not a wall-clock hardware timing proof.
      next=std::max(next+std::chrono::milliseconds(1),started+std::chrono::milliseconds(1));
      std::this_thread::sleep_until(next);
    }
    require(!controller.refused,"ACT target refused; simulation stopped");
#endif
    std::cout << "ACT bridge stopped\n";
    return 0;
  } catch(const std::exception& error) {
    if(link_owner) link_owner->fail(error.what());
    std::cerr << "ACT bridge refused/stopped: " << error.what() << '\n';
#ifdef FR3_ACT_ROBOT
    if(const auto* control=dynamic_cast<const franka::ControlException*>(&error); control && !control->log.empty()) {
      const auto& last=control->log.back().state;
      auto print=[](const char* name,const auto& values) {
        std::cerr << name << "=[";
        for(size_t i=0;i<values.size();++i) std::cerr << (i ? "," : "") << values[i];
        std::cerr << "]\n";
      };
      std::cerr << "Last SDK-logged state before abort (not a new robot read):\n";
      print("q_rad",last.q);
      print("q_d_rad",last.q_d);
      print("dq_d_rad_s",last.dq_d);
      print("ddq_d_rad_s2",last.ddq_d);
      print("measured_joint_torque_Nm",last.tau_J);
      print("joint_external_torque_Nm",last.tau_ext_hat_filtered);
      print("joint_contact",last.joint_contact);
      print("joint_collision",last.joint_collision);
      print("external_wrench_N_Nm",last.O_F_ext_hat_K);
      print("external_wrench_stiffness_frame_N_Nm",last.K_F_ext_hat_K);
      print("cartesian_contact",last.cartesian_contact);
      print("cartesian_collision",last.cartesian_collision);
      std::cerr << "m_ee_kg=" << last.m_ee << ", m_load_kg=" << last.m_load << ", m_total_kg=" << last.m_total << '\n';
      print("load_com_flange_m",last.F_x_Cload);
    }
#endif
    return 1;
  }
}
