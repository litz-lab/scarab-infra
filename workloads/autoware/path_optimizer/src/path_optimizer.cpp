#include <rclcpp/rclcpp.hpp>
#include <rosbag2_interfaces/srv/play_next.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <autoware/path_optimizer/node.hpp>

#include <tier4_planning_msgs/msg/path_with_lane_id.hpp>
#include <autoware_planning_msgs/msg/path.hpp>
#include <autoware_planning_msgs/msg/path_point.hpp>
#include <nav_msgs/msg/odometry.hpp>

using tier4_planning_msgs::msg::PathWithLaneId;
using autoware_planning_msgs::msg::Path;
using autoware_planning_msgs::msg::PathPoint;
using nav_msgs::msg::Odometry;
using PlayNextSrv = rosbag2_interfaces::srv::PlayNext;

class PathOptimizerForTest : public autoware::path_optimizer::PathOptimizer
{
public:
  explicit PathOptimizerForTest(const rclcpp::NodeOptions & opts)
  : autoware::path_optimizer::PathOptimizer(opts) {}

  using autoware::path_optimizer::PathOptimizer::onPath;
};

class POBenchNode : public rclcpp::Node
{
public:
  POBenchNode() : Node("path_optimizer_direct_test")
  {
    rclcpp::NodeOptions opts;
    const auto util_dir = ament_index_cpp::get_package_share_directory("autoware_test_utils");
    const auto po_dir   = ament_index_cpp::get_package_share_directory("autoware_path_optimizer");
    opts.arguments({
      "--ros-args",
      "--params-file", util_dir + "/config/test_vehicle_info.param.yaml",
      "--params-file", util_dir + "/config/test_common.param.yaml",
      "--params-file", util_dir + "/config/test_nearest_search.param.yaml",
      "--params-file", po_dir   + "/config/path_optimizer.param.yaml"
    });
    po_ = std::make_shared<PathOptimizerForTest>(opts);

    sub_path_ = create_subscription<PathWithLaneId>(
      "/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id",
      rclcpp::SensorDataQoS(),
      [this](PathWithLaneId::SharedPtr m){
        last_path_ = std::move(m);
        tryRun();
      });

    sub_odom_ = create_subscription<Odometry>(
      "/localization/kinematic_state", 10,
      [this](Odometry::SharedPtr m){
        last_odom_ = std::move(m);
        tryRun();
      });

    odom_pub_ = create_publisher<Odometry>("path_optimizer/input/odometry", 10);
  }

private:
  void tryRun()
  {
    if (!(last_path_ && last_odom_)) return;

    Path path;
    path.header      = last_path_->header;
    path.left_bound  = last_path_->left_bound;
    path.right_bound = last_path_->right_bound;
    path.points.reserve(last_path_->points.size());
    for (const auto & p_in : last_path_->points) {
      PathPoint p_out;
      p_out.pose                      = p_in.point.pose;
      p_out.longitudinal_velocity_mps = p_in.point.longitudinal_velocity_mps;
      path.points.push_back(p_out);
    }

    Odometry odom = *last_odom_;
    odom.header.stamp = path.header.stamp;
    odom_pub_->publish(odom);

    RCLCPP_INFO(get_logger(), "onPath() invoked (points=%zu)", path.points.size());
    po_->onPath(std::make_shared<const Path>(path));

    last_path_.reset();
  }

  /* members */
  std::shared_ptr<PathOptimizerForTest>           po_;
  rclcpp::Subscription<PathWithLaneId>::SharedPtr sub_path_;
  rclcpp::Subscription<Odometry>::SharedPtr       sub_odom_;
  rclcpp::Publisher<Odometry>::SharedPtr          odom_pub_;

  PathWithLaneId::SharedPtr last_path_;
  Odometry::SharedPtr       last_odom_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto bench    = std::make_shared<POBenchNode>();
  auto play_nd  = rclcpp::Node::make_shared("play_next_client");
  auto play_cli =
    play_nd->create_client<PlayNextSrv>("/rosbag2_player/play_next");

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(bench);
  exec.add_node(play_nd);
  
  while (!play_cli->wait_for_service(std::chrono::seconds(1))) {
    if (!rclcpp::ok()) return 1;
    RCLCPP_INFO(play_nd->get_logger(), "waiting for /play_next …");
  }

  for (int i = 0; i < 2 && rclcpp::ok(); ++i) {
    auto fut = play_cli->async_send_request(std::make_shared<PlayNextSrv::Request>());

    exec.spin_until_future_complete(fut, std::chrono::seconds(15));
  }

  exec.spin_some();

  rclcpp::shutdown();
  return 0;
}