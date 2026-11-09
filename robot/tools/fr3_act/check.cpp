#include "motion.hpp"
#include "collision.hpp"
#include <chrono>
#include <iostream>
#include <memory>

namespace {
using namespace fr3_act;
struct CollisionRobot {
  int calls=0;
  void setCollisionBehavior(const Q& lt_acc,const Q& ut_acc,const Q& lt_nom,const Q& ut_nom,
                            const std::array<double,6>& lf_acc,const std::array<double,6>& uf_acc,
                            const std::array<double,6>& lf_nom,const std::array<double,6>& uf_nom) {
    require(lt_acc==collision_lower_torque_acceleration && ut_acc==collision_upper_torque_acceleration
            && lt_nom==collision_lower_torque_nominal && ut_nom==collision_upper_torque_nominal
            && lf_acc==collision_lower_force_acceleration && uf_acc==collision_upper_force_acceleration
            && lf_nom==collision_lower_force_nominal && uf_nom==collision_upper_force_nominal,
            "Collision SDK argument mapping changed");
    ++calls;
  }
};
void speed_scale_test() {
  const Q initial{0.,0.,0.,-1.5,0.,1.5,0.};
  Q reference_end{};
  for(double scale : {1.,.5,.25,.1}) {
    Motion motion(initial,scale);
    const int count=static_cast<int>(std::lround(1000/scale));
    for(int k=0;k<count;++k) {
      if(k%33==0) {
        Q target=initial, velocity{};
        target[3]-=.6*(k*.001*scale+1./30);
        velocity[3]=-.6*scale;
        require(motion.retarget(target,velocity),"Scaled ramp target rejected");
      }
      const Sample before=motion.state;
      const Sample after=motion.tick(.001);
      require(!motion.refused,"Scaled tracking exceeded braking envelope");
      for(size_t j=0;j<7;++j) {
        require(std::abs(after.v[j])<=velocity_limits[j]*scale+1e-7,"Scaled velocity exceeded");
        require(std::abs(after.a[j])<=acceleration_limits[j]*scale*scale+1e-7,"Scaled acceleration exceeded");
        require(std::abs(after.a[j]-before.a[j])<=jerk_limits[j]*scale*scale*scale*.001+1e-7,
                "Scaled jerk exceeded");
      }
    }
    if(scale==1) reference_end=motion.state.q;
    else require(distance(reference_end,motion.state.q)<.005,"Time dilation changed ramp path");
    require(std::abs(motion.state.q[3]-initial[3])>.5,"Scaled trajectory failed to progress");
    Q excessive_velocity{};
    excessive_velocity[0]=velocity_limits[0]*scale+.01;
    require(!motion.retarget(initial,excessive_velocity),"Scaled velocity boundary accepted invalid target");
    motion.stop();
    for(int k=0;k<50000 && !motion.finished;++k) motion.tick(.001);
    require(motion.finished && distance(motion.state.v,Q{})<1e-8 && distance(motion.state.a,Q{})<1e-8,
            "Scaled motion did not brake to rest");
  }
  for(double bad : {0.,.09,1.01,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}) {
    bool rejected=false;
    try { Motion motion(initial,bad); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"Invalid speed scale accepted");
  }
}
void position_velocity_envelope_test() {
  const Q initial{0.,0.,0.,-1.5,0.,1.5,0.};
  for(double direction : {-1.,1.}) {
    Motion motion(initial);
    Q minimum{}, maximum{}, target=initial;
    minimum.fill(-.3); maximum.fill(.7);
    motion.set_velocity_bounds(minimum,maximum);
    target[0]+=direction*.5;
    require(motion.retarget(target),"Envelope test target refused");
    for(int k=0;k<600;++k) {
      const auto sample=motion.tick(.001);
      require(!motion.refused,"Envelope tracking refused a feasible target");
      for(size_t j=0;j<7;++j)
        require(sample.v[j]>=.95*minimum[j]-1e-7 && sample.v[j]<=.95*maximum[j]+1e-7,
                "Position-dependent directional velocity envelope exceeded");
    }
    require(std::abs(motion.state.q[0]-initial[0])>.1,"Envelope limiter prevented motion");
    motion.stop();
    for(int k=0;k<5000 && !motion.finished;++k) motion.tick(.001);
    require(motion.finished,"Envelope motion failed to brake");
    maximum[0]=std::numeric_limits<double>::quiet_NaN();
    bool rejected=false;
    try { motion.set_velocity_bounds(minimum,maximum); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"Nonfinite SDK velocity bounds accepted");
  }
}
void self_test() {
  position_velocity_envelope_test();
  speed_scale_test();
  {
    Q initial{0.,0.,0.,-1.5,0.,1.5,0.}, forward=initial, reverse=initial;
    forward[0]=.2; reverse[0]=-.2;
    Motion reversal(initial);
    require(reversal.retarget(forward),"Forward target refused");
    for(int k=0;k<150;++k) reversal.tick(.001);
    require(reversal.state.v[0]>0,"Reversal test did not establish forward motion");
    const auto before=reversal.state;
    require(reversal.retarget(reverse),"Reverse target refused");
    auto after=reversal.tick(.001);
    require(after.v[0]>0 && after.q[0]>before.q[0],"Direction reversed without braking");
    require(std::abs(after.a[0]-before.a[0])<=jerk_limits[0]*.001+1e-7,
            "Direction change exceeded jerk limit");
    reversal.stop();
    for(int k=0;k<2000 && !reversal.finished;++k) reversal.tick(.001);
    require(reversal.finished && !reversal.retarget(initial),"Stopped stream resumed old targets");
    Motion rearmed(initial);
    require(distance(rearmed.state.v,Q{})==0 && distance(rearmed.state.a,Q{})==0,
            "New session inherited previous motion history");
  }
  CollisionRobot collision_robot;
  configure_collision(collision_robot);
  require(collision_robot.calls==1,"Collision configuration was not sent exactly once");
  for(double bad : {-1.,0.,std::numeric_limits<double>::quiet_NaN()}) {
    bool refused=false;
    try { check_collision_pair(std::array<double,1>{20},std::array<double,1>{bad}); }
    catch(const std::runtime_error&) { refused=true; }
    require(refused,"Invalid collision threshold accepted");
  }
  Q q{0.,0.,0.,-1.15,0.,1.5,0.};
  Motion motion(q);
  double peak=0, error=0;
  for(int k=0;k<4000;++k) {
    if(k%33==0) {
      const double t=(k+33)*0.001;
      Q target=q, velocity{};
      target[3]=-1.6+0.45*std::cos(4.2*t);
      velocity[3]=-0.45*4.2*std::sin(4.2*t);
      Sample previous=motion.state;
      require(motion.retarget(target,velocity),"Continuous target rejected");
      require(distance(previous.q,motion.state.q)==0 && distance(previous.v,motion.state.v)==0
              && distance(previous.a,motion.state.a)==0,"Retarget reset the motion state");
    }
    const auto state=motion.tick(.001);
    peak=std::max(peak,std::abs(state.v[3]));
    error=std::max(error,std::abs(state.q[3]-(-1.6+0.45*std::cos(4.2*(k+1)*.001))));
    require(!motion.refused,"Valid continuous trajectory refused");
  }
  require(peak>1.5,"Dataset-speed motion was still slowed to commissioning speed");
  require(error<0.10,"Continuous tracking lag exceeds 0.10 rad");
  motion.stop();
  for(int k=0;k<2000 && !motion.finished;++k) motion.tick(.001);
  require(motion.finished && distance(motion.state.v,Q{})<1e-8 && distance(motion.state.a,Q{})<1e-8,
          "Stop did not reach zero velocity and acceleration");
  require(!motion.retarget(q),"Stopped motion accepted a restart");
  Motion invalid(q);
  Q bad=q; bad[0]=100;
  require(!invalid.retarget(bad),"Out-of-range target accepted");
  bad=q; bad[0]=std::numeric_limits<double>::quiet_NaN();
  require(!invalid.retarget(bad),"NaN target accepted");
  Q near=upper; near[0]-=.001;
  Motion boundary(near);
  Q outward{}; outward[0]=velocity_limits[0];
  if(boundary.retarget(upper,outward)) {
    for(int k=0;k<2000 && !boundary.finished;++k) boundary.tick(.001);
    require(boundary.stopping,"Motion crossed its braking envelope");
  }
  std::cout << "PASS: Ruckig tracking peak=" << peak << " rad/s, max error=" << error
            << " rad; continuous q/v/a, derivative limits, bounded stop, NaN and boundary rejection\n";
}
void dataset_check(const char* filename, double speed_scale=1.0) {
  const MotionLimits limits(speed_scale);
  std::ifstream file(filename);
  require(file.good(),"Cannot open dataset check file");
  std::unique_ptr<Motion> motion;
  int episode=-1, previous=-1;
  size_t frames=0, refused=0, guard=0, ticks=0;
  Q q{}, target{}, velocity{}, peak_v{}, peak_a{}, peak_j{}, peak_error{};
  const auto begin=std::chrono::steady_clock::now();
  while(file>>episode) {
    for(double& x:q) require(static_cast<bool>(file>>x),"Missing state");
    for(double& x:target) require(static_cast<bool>(file>>x),"Missing target");
    for(double& x:velocity) require(static_cast<bool>(file>>x),"Missing target velocity");
    if(episode!=previous) {
      motion=std::make_unique<Motion>(q,speed_scale); previous=episode;
    }
    for(size_t j=0;j<7;++j) {
      velocity[j]=std::clamp(velocity[j]*speed_scale,-limits.velocity[j],limits.velocity[j]);
      if(std::abs(target[j]-motion->state.q[j])>(1.25*std::max(.10,.05+.075*velocity_limits[j]))) ++guard;
    }
    if(!motion->retarget(target,velocity)) ++refused;
    const int count=static_cast<int>(std::lround((frames+1)*1000./(30*speed_scale))
                                    -std::lround(frames*1000./(30*speed_scale)));
    for(int k=0;k<count;++k) {
      const auto before=motion->state;
      const auto state=motion->tick(.001);
      for(size_t j=0;j<7;++j) {
        peak_v[j]=std::max(peak_v[j],std::abs(state.v[j]));
        peak_a[j]=std::max(peak_a[j],std::abs(state.a[j]));
        peak_j[j]=std::max(peak_j[j],std::abs(state.a[j]-before.a[j])/.001);
      }
      ++ticks;
    }
    for(size_t j=0;j<7;++j) peak_error[j]=std::max(peak_error[j],std::abs(motion->state.q[j]-target[j]));
    ++frames;
  }
  require(file.eof() && frames>0,"Invalid/empty dataset check file");
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
  auto print=[](const Q& x) {
    std::cout << '[';
    for(size_t j=0;j<7;++j) std::cout << (j ? "," : "") << x[j];
    std::cout << ']';
  };
  std::cout << "{\"speed_scale\":" << speed_scale << ",\"frames\":" << frames << ",\"simulation_ticks\":" << ticks
            << ",\"refused_targets\":" << refused << ",\"goal_guard_exceedances\":" << guard
            << ",\"compute_seconds\":" << seconds << ",\"peak_velocity\":";
  print(peak_v); std::cout << ",\"peak_acceleration\":"; print(peak_a);
  std::cout << ",\"peak_jerk\":"; print(peak_j);
  std::cout << ",\"max_tracking_error\":"; print(peak_error);
  std::cout << "}\n";
  require(refused==0 && guard==0,"Dataset replay exceeded online feasibility or target guard");
}
}
int main(int argc,char** argv) {
  try {
    if(argc==3 && std::string(argv[1])=="--dataset") dataset_check(argv[2]);
    else if(argc==5 && std::string(argv[1])=="--dataset" && std::string(argv[3])=="--speed-scale")
      dataset_check(argv[2],std::stod(argv[4]));
    else { fr3_act::require(argc==1,"Usage: fr3_act_check [--dataset FILE [--speed-scale S]]"); self_test(); }
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
