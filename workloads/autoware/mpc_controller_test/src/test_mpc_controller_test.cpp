#include <rosbag2_interfaces/srv/play_next.hpp> 

#include <rclcpp/rclcpp.hpp>

#include "autoware/mpc_lateral_controller/mpc.hpp"
#include "autoware/mpc_lateral_controller/qp_solver/qp_solver_unconstraint_fast.hpp"
#include "autoware/mpc_lateral_controller/vehicle_model/vehicle_model_bicycle_kinematics.hpp"

#include "autoware_control_msgs/msg/lateral.hpp"
#include "autoware_internal_debug_msgs/msg/float32_multi_array_stamped.hpp"
#include "autoware_planning_msgs/msg/trajectory.hpp"
#include "autoware_vehicle_msgs/msg/steering_report.hpp"
#include "nav_msgs/msg/odometry.hpp"

using namespace autoware::motion::control::mpc_lateral_controller;
using autoware_control_msgs::msg::Lateral;
using autoware_internal_debug_msgs::msg::Float32MultiArrayStamped;
using autoware_planning_msgs::msg::Trajectory;
using autoware_vehicle_msgs::msg::SteeringReport;
using nav_msgs::msg::Odometry;
using autoware::motion::control::trajectory_follower::LateralHorizon;
using PlayNextSrv = rosbag2_interfaces::srv::PlayNext;

class MPCRunner
{
public:
  explicit MPCRunner(const rclcpp::Node::SharedPtr & node)
  : node_(node), logger_(node_->get_logger())
  {
    mpc_ = std::make_unique<MPC>(*node_);
    mpc_->setLogger(logger_);
    mpc_->setClock(node_->get_clock());
    mpc_->initializeSteeringPredictor();

    mpc_->m_ctrl_period = 0.03;
    mpc_->m_steer_rate_lim_map_by_curvature = {{0.0, 2.0}, {9999.0, 2.0}};
    mpc_->m_steer_rate_lim_map_by_velocity  = {{0.0, 2.0}, {9999.0, 2.0}};
    mpc_->setVehicleModel(std::make_shared<KinematicsBicycleModel>(2.7, 0.610865, 0.1));
    mpc_->setQPSolver(std::make_shared<QPSolverEigenLeastSquareLLT>());

    auto & p = mpc_->m_param;
    p.prediction_horizon = 8;
    p.prediction_dt = 0.1;
    p.zero_ff_steer_deg = 1.0;
    p.min_prediction_length = 2.0;
    p.acceleration_limit = 2.0;
    p.velocity_time_constant = 0.3;
   
    p.nominal_weight.lat_error = 1.0;
    p.nominal_weight.heading_error = 1.0;
    p.nominal_weight.heading_error_squared_vel = 1.0;
    p.nominal_weight.terminal_lat_error = 1.0;
    p.nominal_weight.terminal_heading_error = 0.1;
  
    p.nominal_weight.steering_input = 1.0;
    p.nominal_weight.steering_input_squared_vel = 0.25;
    p.nominal_weight.lat_jerk = 0.0;
    p.nominal_weight.steer_rate = 0.0;
    p.nominal_weight.steer_acc = 0.000001;
  
    p.low_curvature_weight.lat_error = 0.1;
    p.low_curvature_weight.heading_error = 0.0;
    p.low_curvature_weight.heading_error_squared_vel = 0.3;
    p.low_curvature_weight.steering_input = 1.0;
    p.low_curvature_weight.steering_input_squared_vel = 0.25;
    p.low_curvature_weight.lat_jerk = 0.0;
    p.low_curvature_weight.steer_rate = 0.0;
    p.low_curvature_weight.steer_acc = 0.000001;
  }

  bool run(const Trajectory & ref_traj, const Odometry & odom, const SteeringReport & steer)
  {
    TrajectoryFilteringParam traj_filter;
    traj_filter.traj_resample_dist = 0.1;
    traj_filter.enable_path_smoothing = false;
    traj_filter.path_filter_moving_ave_num = 5;
    traj_filter.curvature_smoothing_num_traj = 3;
    traj_filter.curvature_smoothing_num_ref_steer = 3;
    traj_filter.extend_trajectory_for_end_yaw_control = false;

    mpc_->setReferenceTrajectory(ref_traj, traj_filter, odom);

    Lateral cmd; 
    Trajectory pred; 
    Float32MultiArrayStamped diag; 
    LateralHorizon hz;

    auto res = mpc_->calculateMPC(steer, odom, cmd, pred, diag, hz);

    if (!res.result) {
      RCLCPP_ERROR(logger_, "MPC failed: %s", res.reason.c_str());
      return false;
    }

    RCLCPP_INFO(logger_, "steer = %.6f  rate = %.6f", cmd.steering_tire_angle, cmd.steering_tire_rotation_rate);
    RCLCPP_INFO(logger_, "input_steer steer = %.6f", steer.steering_tire_angle);

    return true;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger          logger_;
  std::unique_ptr<MPC>    mpc_;
};

class MPCTestNode : public rclcpp::Node
{
public:
  bool isFinished() const { return executed_; }
  explicit MPCTestNode() : Node("mpc_test_node") {}

  void initRunner(const std::shared_ptr<MPCRunner> & runner)
  {
    runner_ = runner;

    odom_sub_ = create_subscription<Odometry>(
      "/localization/kinematic_state", 10,
      [this](Odometry::SharedPtr msg) { odom_ = std::move(msg); tryRun(); });

    steer_sub_ = create_subscription<SteeringReport>(
      "/vehicle/status/steering_status", 10,
      [this](SteeringReport::SharedPtr msg) { steer_ = std::move(msg); tryRun(); });

    traj_sub_ = create_subscription<Trajectory>(
      "/planning/scenario_planning/lane_driving/trajectory", 10,
      [this](Trajectory::SharedPtr msg) { traj_ = std::move(msg); tryRun(); });
  }

private:
  void tryRun()
  {
    if (executed_ || !runner_) return;
    if (odom_ && steer_ && traj_) {
      executed_ = true;
      bool ok = runner_->run(*traj_, *odom_, *steer_);
      RCLCPP_INFO(get_logger(), "MPC test %s", ok ? "passed" : "failed");
      rclcpp::shutdown();
    }
  }

  rclcpp::Subscription<Odometry>::SharedPtr       odom_sub_;
  rclcpp::Subscription<SteeringReport>::SharedPtr steer_sub_;
  rclcpp::Subscription<Trajectory>::SharedPtr     traj_sub_;
  Odometry::SharedPtr       odom_;
  SteeringReport::SharedPtr steer_;
  Trajectory::SharedPtr     traj_;

  std::shared_ptr<MPCRunner> runner_;
  bool executed_{false};
};

int main(int argc, char ** argv)
{
  try {
    rclcpp::init(argc, argv);

    auto mpc_node = std::make_shared<MPCTestNode>();
    auto runner   = std::make_shared<MPCRunner>(mpc_node);
    mpc_node->initRunner(runner);

    auto play_cli = mpc_node->create_client<PlayNextSrv>("/rosbag2_player/play_next");

    while (!play_cli->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_INFO(mpc_node->get_logger(), "Waiting for /rosbag2_player/play_next …");
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(mpc_node->get_logger(), "Interrupted while waiting.");
        rclcpp::shutdown();
        return 1;
      }
    }

    for (int i = 1; i <= 3; ++i) {
      RCLCPP_INFO(mpc_node->get_logger(), ">> play_next() call #%d", i);
      auto req  = std::make_shared<PlayNextSrv::Request>();
      auto fut  = play_cli->async_send_request(req);
    
      auto stat = rclcpp::spin_until_future_complete(mpc_node, fut, std::chrono::seconds(15));
    
      if (mpc_node->isFinished()) break;
    
      if (stat == rclcpp::FutureReturnCode::SUCCESS) {
        RCLCPP_INFO(mpc_node->get_logger(), "play_next #%d succeeded", i);
      } else {
        RCLCPP_ERROR(mpc_node->get_logger(), "play_next #%d TIMEOUT / FAILED", i);
        break;
      }
    }

    while (rclcpp::ok() && !mpc_node->isFinished()) {
      rclcpp::spin_some(mpc_node);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    rclcpp::shutdown();
  } catch (const std::exception & e) {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
