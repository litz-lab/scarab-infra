#include <rclcpp/rclcpp.hpp>

#include <autoware/path_optimizer/node.hpp>
#include <autoware_planning_test_manager/autoware_planning_test_manager.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <tier4_planning_msgs/msg/path_with_lane_id.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "rosbag2_interfaces/srv/play_next.hpp"

#include <memory>

class PathOptimizerManualTest : public rclcpp::Node
{
public:
  std::shared_ptr<autoware::planning_test_manager::PlanningInterfaceTestManager> test_manager_;
  std::shared_ptr<autoware::path_optimizer::PathOptimizer>                       test_target_node_;
  rclcpp::Subscription<tier4_planning_msgs::msg::PathWithLaneId>::SharedPtr      sub_path_with_lane_id_;

  explicit PathOptimizerManualTest(const rclcpp::NodeOptions & options)
  : Node("path_optimizer_manual_test", options)
  {
    test_manager_ = std::make_shared<autoware::planning_test_manager::PlanningInterfaceTestManager>();

    rclcpp::NodeOptions po_opts;
    const auto utils_dir = ament_index_cpp::get_package_share_directory("autoware_test_utils");
    const auto po_dir    = ament_index_cpp::get_package_share_directory("autoware_path_optimizer");

    po_opts.arguments({
      "--ros-args",
      "--params-file", utils_dir + "/config/test_vehicle_info.param.yaml",
      "--params-file", utils_dir + "/config/test_common.param.yaml",
      "--params-file", utils_dir + "/config/test_nearest_search.param.yaml",
      "--params-file", po_dir   + "/config/path_optimizer.param.yaml"
    });

    test_target_node_ = std::make_shared<autoware::path_optimizer::PathOptimizer>(po_opts);

    test_manager_->publishOdometry(test_target_node_, "path_optimizer/input/odometry");
    test_manager_->setTrajectorySubscriber("path_optimizer/output/path");
    test_manager_->setPathInputTopicName("path_optimizer/input/path");

    sub_path_with_lane_id_ =
      create_subscription<tier4_planning_msgs::msg::PathWithLaneId>(
        "/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id",
        rclcpp::SensorDataQoS(),
        std::bind(&PathOptimizerManualTest::onExternalPathWithLaneId, this, std::placeholders::_1));
  }

  void onExternalPathWithLaneId(
    const tier4_planning_msgs::msg::PathWithLaneId::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(), "Received external PathWithLaneId: points = %zu", msg->points.size());
    RCLCPP_INFO(get_logger(), "initial count = %d", test_manager_->getReceivedTopicNum());

    test_manager_->testWithArbitraryPath(test_target_node_, *msg);

    if (test_manager_->getReceivedTopicNum() != 0)
        RCLCPP_INFO(get_logger(), "[PASS] received count = %d", test_manager_->getReceivedTopicNum());
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PathOptimizerManualTest>(rclcpp::NodeOptions{});
  auto play_node = rclcpp::Node::make_shared("combined_client_node");

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  exec.add_node(play_node);

  auto client = play_node->create_client<rosbag2_interfaces::srv::PlayNext>("/rosbag2_player/play_next");
  client->wait_for_service();
  auto future = client->async_send_request(std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>());

  exec.spin_until_future_complete(future);
  exec.cancel();
  
  rclcpp::shutdown();
  return 0;
}
