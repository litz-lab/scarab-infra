#include <rclcpp/rclcpp.hpp>

#include "autoware/mpc_lateral_controller/mpc.hpp"
#include "autoware/mpc_lateral_controller/qp_solver/qp_solver_osqp.hpp"
#include "autoware/mpc_lateral_controller/qp_solver/qp_solver_unconstraint_fast.hpp"
#include "autoware/mpc_lateral_controller/vehicle_model/vehicle_model_bicycle_dynamics.hpp"
#include "autoware/mpc_lateral_controller/vehicle_model/vehicle_model_bicycle_kinematics.hpp"
#include "autoware/mpc_lateral_controller/vehicle_model/vehicle_model_bicycle_kinematics_no_delay.hpp"

#include <autoware/trajectory_follower_base/control_horizon.hpp>

#include "autoware_control_msgs/msg/lateral.hpp"
#include "autoware_internal_debug_msgs/msg/float32_multi_array_stamped.hpp"
#include "autoware_planning_msgs/msg/trajectory.hpp"
#include "autoware_planning_msgs/msg/trajectory_point.hpp"
#include "autoware_vehicle_msgs/msg/steering_report.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/odometry.hpp"

#include "autoware/motion_utils/trajectory/trajectory.hpp"
#include "autoware/mpc_lateral_controller/mpc_utils.hpp"

#include <chrono>
#include <random>

using namespace autoware::motion::control::mpc_lateral_controller;

using autoware_control_msgs::msg::Lateral;
using autoware_internal_debug_msgs::msg::Float32MultiArrayStamped;
using autoware_planning_msgs::msg::Trajectory;
using autoware_planning_msgs::msg::TrajectoryPoint;
using autoware_vehicle_msgs::msg::SteeringReport;
using nav_msgs::msg::Odometry;
 
using autoware::motion::control::trajectory_follower::LateralHorizon;

TrajectoryPoint makePoint(const double x, const double y, const float vx)
{
  TrajectoryPoint p;
  p.pose.position.x = x;
  p.pose.position.y = y;
  p.longitudinal_velocity_mps = vx;
  return p;
}
 
int main(int argc, char ** argv)
{
  auto start_time = std::chrono::steady_clock::now();
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("test_mpc_controller_test_node");

  auto mpc = std::make_unique<MPC>(*node);
  mpc->setLogger(node->get_logger());
  mpc->setClock(node->get_clock());
  mpc->initializeSteeringPredictor();

  mpc->m_ctrl_period = 0.03;

  mpc->m_steer_rate_lim_map_by_curvature.emplace_back(0.0, 2.0);
  mpc->m_steer_rate_lim_map_by_curvature.emplace_back(9999.0, 2.0);
  mpc->m_steer_rate_lim_map_by_velocity.emplace_back(0.0, 2.0);
  mpc->m_steer_rate_lim_map_by_velocity.emplace_back(9999.0, 2.0);
 
  auto vehicle_model_ptr = std::make_shared<KinematicsBicycleModel>(2.7, 0.610865, 0.1);
  mpc->setVehicleModel(vehicle_model_ptr);
 
  auto qpsolver_ptr = std::make_shared<QPSolverEigenLeastSquareLLT>();
  mpc->setQPSolver(qpsolver_ptr);

  auto & p = mpc->m_param;
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

  TrajectoryFilteringParam traj_filter;
  traj_filter.traj_resample_dist = 0.1;
  traj_filter.enable_path_smoothing = false;
  traj_filter.path_filter_moving_ave_num = 5;
  traj_filter.curvature_smoothing_num_traj = 3;
  traj_filter.curvature_smoothing_num_ref_steer = 3;
  traj_filter.extend_trajectory_for_end_yaw_control = false;

  std::random_device rd;
  std::mt19937 gen(rd());

  double max_step = 3.0;
  std::uniform_real_distribution<double> dist_step(-max_step, max_step);

  Trajectory dummy_trajectory;
  int num_points = 10;
  double y_current = 0.0;
  for (int i = 0; i < num_points; ++i) {
    double x = static_cast<double>(i);
    double step = dist_step(gen);
    y_current += step;

    dummy_trajectory.points.push_back(makePoint(x, y_current, 1.0f));
  }

  SteeringReport neutral_steer;
  neutral_steer.steering_tire_angle = 0.0F;
 
  Odometry odom;
  odom.pose.pose.position.x = 0.0;
  odom.pose.pose.position.y = 0.0;
  odom.twist.twist.linear.x = 1.0;
 
  mpc->setReferenceTrajectory(dummy_trajectory, traj_filter, odom);
 
  Lateral ctrl_cmd;
  Trajectory pred_traj;
  Float32MultiArrayStamped diag;
  LateralHorizon ctrl_cmd_horizon;
 
  const auto result =
    mpc->calculateMPC(neutral_steer, odom, ctrl_cmd, pred_traj, diag, ctrl_cmd_horizon);
 
  if (!result.result) {
    RCLCPP_ERROR(node->get_logger(), "MPC calculation failed: %s", result.reason.c_str());
    rclcpp::shutdown();
    return 1;
  }

  double eps = 0.0f;
  bool steer_ok = (ctrl_cmd.steering_tire_angle < eps);
  bool steer_rate_ok = (ctrl_cmd.steering_tire_rotation_rate < eps);
 
  RCLCPP_INFO(
    node->get_logger(),
    "Calculated steer=%.6f, steer_rate=%.6f",
    ctrl_cmd.steering_tire_angle,
    ctrl_cmd.steering_tire_rotation_rate
  );

  RCLCPP_INFO(node->get_logger(), "MPC test passed successfully!");
  rclcpp::shutdown();

  auto end_time = std::chrono::steady_clock::now();
  auto duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

  RCLCPP_INFO(node->get_logger(), "Execution time: %.3f ms", duration_ms);

  return 0;
}