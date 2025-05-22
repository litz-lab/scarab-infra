#include <memory>
#include <atomic>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "rosbag2_interfaces/srv/play_next.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "autoware/euclidean_cluster/voxel_grid_based_euclidean_cluster.hpp"
#include "tier4_perception_msgs/msg/detected_objects_with_feature.hpp"

namespace cluster_test
{

class ClusterTestNode : public rclcpp::Node
{
public:
  std::atomic_bool frame_done_;

  explicit ClusterTestNode(const rclcpp::NodeOptions & opts = rclcpp::NodeOptions())
  : Node("cluster_test_node", opts), frame_done_(false)
  {
    using namespace std::placeholders;

    const bool  use_height                  = false;
    const int   min_cluster_size            = 5;
    const int   max_cluster_size            = 8000;
    const float tolerance                   = 0.5;
    const float voxel_leaf_size             = 0.2;
    const int   min_points_number_per_voxel = 1;
    cluster_ = std::make_shared<
      autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster>(
        use_height, min_cluster_size, max_cluster_size,
        tolerance, voxel_leaf_size, min_points_number_per_voxel);

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/sensing/lidar/top/rectified/pointcloud",
      rclcpp::SensorDataQoS(),
      std::bind(&ClusterTestNode::cloudCallback, this, _1));

    RCLCPP_INFO(get_logger(), "ClusterTestNode ready - waiting for one PointCloud2 frame…");
  }

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
  {
    tier4_perception_msgs::msg::DetectedObjectsWithFeature out;
    if (!cluster_->cluster(msg, out)) {
        RCLCPP_ERROR(get_logger(), "Cluster routine returned false");
        rclcpp::shutdown();
        return;
    }

    const auto n = out.feature_objects.size();
    //RCLCPP_INFO(get_logger(), "Received frame: %zu clusters detected", n);

    if (n == 0) {
        RCLCPP_ERROR(get_logger(), "Test FAILED: cluster count == 0");
    } else {
        //RCLCPP_INFO(get_logger(), "Test PASSED");
    }

    frame_done_.store(true, std::memory_order_release);
    rclcpp::shutdown();
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  std::shared_ptr<
    autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster> cluster_;
};

}  // namespace cluster_test

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto test_node  = std::make_shared<cluster_test::ClusterTestNode>();
  auto client_node = rclcpp::Node::make_shared("play_next_client");

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(test_node);
  exec.add_node(client_node);

  auto client = client_node->create_client<rosbag2_interfaces::srv::PlayNext>(
    "/rosbag2_player/play_next");
  client->wait_for_service();
  client->async_send_request(std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>());

  exec.spin();
  rclcpp::shutdown();
  return 0;
}