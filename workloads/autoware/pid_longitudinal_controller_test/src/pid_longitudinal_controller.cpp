#include <rclcpp/rclcpp.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "autoware/pid_longitudinal_controller/pid_longitudinal_controller.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include "autoware_planning_msgs/msg/trajectory.hpp"
#include "autoware_adapi_v1_msgs/msg/operation_mode_state.hpp"
#include "rosbag2_interfaces/srv/play_next.hpp"

using autoware::motion::control::pid_longitudinal_controller::PidLongitudinalController;
using autoware::motion::control::trajectory_follower::InputData;
using autoware::motion::control::trajectory_follower::LongitudinalOutput;

using nav_msgs::msg::Odometry;
using geometry_msgs::msg::AccelWithCovarianceStamped;
using autoware_planning_msgs::msg::Trajectory;
using autoware_adapi_v1_msgs::msg::OperationModeState;
using PlayNextSrv = rosbag2_interfaces::srv::PlayNext;

rclcpp::NodeOptions makeNodeOptions(
  const bool enable_keep_stopped_until_steer_convergence = false)
{
  const auto share_dir =
    ament_index_cpp::get_package_share_directory("autoware_trajectory_follower_node");
  const auto longitudinal_share_dir =
    ament_index_cpp::get_package_share_directory("autoware_pid_longitudinal_controller");
  const auto lateral_share_dir =
    ament_index_cpp::get_package_share_directory("autoware_mpc_lateral_controller");

  rclcpp::NodeOptions opts;
  opts.append_parameter_override("lateral_controller_mode", "mpc");
  opts.append_parameter_override("longitudinal_controller_mode", "pid");
  opts.append_parameter_override(
    "enable_keep_stopped_until_steer_convergence",
    enable_keep_stopped_until_steer_convergence);

  opts.arguments({
    "--ros-args", "--params-file",
    lateral_share_dir + "/param/lateral_controller_defaults.param.yaml",
    "--params-file",
    longitudinal_share_dir + "/config/autoware_pid_longitudinal_controller.param.yaml",
    "--params-file",
    share_dir + "/test/test_vehicle_info.param.yaml",
    "--params-file",
    share_dir + "/test/test_nearest_search.param.yaml",
    "--params-file",
    share_dir + "/param/trajectory_follower_node.param.yaml"});

  return opts;
}

class LongitudinalRunner
{
public:
  explicit LongitudinalRunner(const rclcpp::Node::SharedPtr & node)
  : node_(node), logger_(node_->get_logger())
  {
    node_->declare_parameter<double>("ctrl_period", 0.03);

    node_->declare_parameter<double>("wheel_radius",         0.39);  // m
    node_->declare_parameter<double>("wheel_width",        0.42);  // m
    node_->declare_parameter<double>("wheel_base",         2.74);  // m
    node_->declare_parameter<double>("wheel_tread",        1.63);  // m
    node_->declare_parameter<double>("front_overhang",     1.00);  // m
    node_->declare_parameter<double>("rear_overhang",      1.03);  // m
    node_->declare_parameter<double>("left_overhang",      0.10);  // m
    node_->declare_parameter<double>("right_overhang",     0.10);  // m
    node_->declare_parameter<double>("vehicle_height",     2.50);  // m
    node_->declare_parameter<double>("max_steer_angle",  0.70);  // rad

    diag_updater_ = std::make_shared<diagnostic_updater::Updater>(node_);
    diag_updater_->setHardwareID("test-node"); 

    controller_   = std::make_unique<PidLongitudinalController>(*node_, diag_updater_);
  }

  bool run(const Trajectory & traj,
           const Odometry   & odom,
           const AccelWithCovarianceStamped & accel)
  {
    InputData in;
    in.current_trajectory = traj;
    in.current_odometry   = odom;
    in.current_accel      = accel;

    OperationModeState op;
    op.mode = OperationModeState::AUTONOMOUS;
    op.is_autoware_control_enabled = true;
    in.current_operation_mode = op;

    LongitudinalOutput out = controller_->run(in);

    RCLCPP_INFO(logger_, "Cmd vel=%.3f  acc=%.3f", out.control_cmd.velocity, out.control_cmd.acceleration);
    return true;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger          logger_;
  std::shared_ptr<diagnostic_updater::Updater> diag_updater_;
  std::unique_ptr<PidLongitudinalController>   controller_;
};

class LongitudinalTestNode : public rclcpp::Node
{
public:
  LongitudinalTestNode() : Node("pid_longitudinal_controller_test_node") {}
  explicit LongitudinalTestNode(const rclcpp::NodeOptions & opts) : rclcpp::Node("pid_longitudinal_controller_test_node", opts) {}

  bool isFinished() const { return executed_; }

  void initRunner(const std::shared_ptr<LongitudinalRunner> & runner)
  {
    runner_ = runner;

    odom_sub_ = create_subscription<Odometry>(
      "/localization/kinematic_state", 10,
      [this](Odometry::SharedPtr msg) { odom_ = std::move(msg); tryRun(); });

    accel_sub_ = create_subscription<AccelWithCovarianceStamped>(
      "/localization/acceleration", 10,
      [this](AccelWithCovarianceStamped::SharedPtr msg) { accel_ = std::move(msg); tryRun(); });

    traj_sub_ = create_subscription<Trajectory>(
      "/planning/scenario_planning/lane_driving/trajectory", 10,
      [this](Trajectory::SharedPtr msg) { traj_ = std::move(msg); tryRun(); });
  }

private:
  void tryRun()
  {
    if (executed_ || !runner_) return;
    if (odom_ && accel_ && traj_) {
      executed_ = true;
      const bool ok = runner_->run(*traj_, *odom_, *accel_);
      RCLCPP_INFO(get_logger(), "PID-longitudinal test %s", ok ? "passed" : "failed");
      rclcpp::shutdown();
    }
  }

  std::shared_ptr<LongitudinalRunner> runner_;
  bool executed_{false};

  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<AccelWithCovarianceStamped>::SharedPtr accel_sub_;
  rclcpp::Subscription<Trajectory>::SharedPtr traj_sub_;
  Odometry::SharedPtr odom_;
  AccelWithCovarianceStamped::SharedPtr accel_;
  Trajectory::SharedPtr traj_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node   = std::make_shared<LongitudinalTestNode>(makeNodeOptions());
  auto runner = std::make_shared<LongitudinalRunner>(node);
  node->initRunner(runner);

  auto play_cli = node->create_client<PlayNextSrv>("/rosbag2_player/play_next");
  while (!play_cli->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_INFO(node->get_logger(), "waiting for /rosbag2_player/play_next …");
    if (!rclcpp::ok()) return 1;
  }

  for (int i = 1; i <= 3 && rclcpp::ok(); ++i) {
    auto fut = play_cli->async_send_request(std::make_shared<PlayNextSrv::Request>());
    if (rclcpp::spin_until_future_complete(node, fut,
        std::chrono::seconds(15)) != rclcpp::FutureReturnCode::SUCCESS) break;
    if (node->isFinished()) break;
  }

  while (rclcpp::ok() && !node->isFinished()) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  rclcpp::shutdown();
  return 0;
}