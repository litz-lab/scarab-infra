#include <memory>
#include <string>
#include <atomic>
#include <cstdint>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tier4_perception_msgs/msg/detected_objects_with_feature.hpp"
#include "autoware/euclidean_cluster/voxel_grid_based_euclidean_cluster.hpp"

#include "rosbag2_storage/storage_options.hpp"
#include "rosbag2_cpp/reader.hpp"
#include "rclcpp/serialization.hpp"
#include "rclcpp/serialized_message.hpp"

namespace cluster_test
{

class ClusterTestNode : public rclcpp::Node
{
public:
  ClusterTestNode(uint64_t start_stamp,
                  const rclcpp::NodeOptions & opts = rclcpp::NodeOptions())
  : Node("cluster_test_node_offline", opts),
    start_stamp_(start_stamp),
    frame_done_(false)
  {
    const bool  use_height                  = false;
    const int   min_cluster_size            = 1;
    const int   max_cluster_size            = 500;
    const float tolerance                   = 1.0f;
    const float voxel_leaf_size             = 0.5f;
    const int   min_points_number_per_voxel = 3;

    cluster_ = std::make_shared<
      autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster>(
        use_height, min_cluster_size, max_cluster_size,
        tolerance, voxel_leaf_size, min_points_number_per_voxel);

    RCLCPP_INFO(get_logger(), "bag: %s, start_stamp=%lu", BAG_PATH,
                static_cast<unsigned long>(start_stamp_));
  }

  bool run()
  {
    using rosbag2_storage::StorageOptions;

    rosbag2_cpp::Reader reader;
    StorageOptions opts;
    opts.uri = BAG_PATH;
    opts.storage_id = "sqlite3";

    reader.open(opts);

    if (start_stamp_ > 0) {
      try {
        reader.seek(start_stamp_ + 1);
      } catch (const std::exception & e) {
        RCLCPP_WARN(get_logger(), "seek() failed: %s — starting from beginning", e.what());
      }
    }

    rclcpp::Serialization<sensor_msgs::msg::PointCloud2> ser;

    while (reader.has_next()) {
      auto bag_msg = reader.read_next();
      if (bag_msg->topic_name != "/sensing/lidar/top/rectified/pointcloud") {
        continue;  // skip unrelated topics
      }

      rclcpp::SerializedMessage serialized(*bag_msg->serialized_data);
      auto pc2 = std::make_shared<sensor_msgs::msg::PointCloud2>();
      ser.deserialize_message(&serialized, pc2.get());

      tier4_perception_msgs::msg::DetectedObjectsWithFeature out;
      for (int i = 0; i < 500; i++) {
        if (!cluster_->cluster(pc2, out)) {
          RCLCPP_ERROR(get_logger(), "Cluster routine returned false");
          return false;
        }
      }

      const auto n = out.feature_objects.size();
      if (n == 0) {
        RCLCPP_ERROR(get_logger(), "Test FAILED: cluster count == 0");
      } else {
        RCLCPP_INFO(get_logger(), "Test PASSED: %zu clusters detected", n);
      }

      // Print timestamp so caller can capture it for next run
      std::cout << bag_msg->time_stamp << std::endl;
      frame_done_.store(true);
      return n > 0;
    }

    RCLCPP_ERROR(get_logger(), "No suitable messages found in bag");
    return false;
  }

private:
  static constexpr const char * BAG_PATH = "/tmp_home/autoware/euclidean_cluster/input_bag";
  uint64_t start_stamp_;
  std::atomic_bool frame_done_;
  std::shared_ptr<
    autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster> cluster_;
};

}  // namespace cluster_test

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  uint64_t start_stamp = 1649926360285939385;
  if (argc > 1) {
    start_stamp = std::stoull(argv[1]);
  }

  auto node = std::make_shared<cluster_test::ClusterTestNode>(start_stamp);
  bool ok = node->run();

  rclcpp::shutdown();
  return ok ? 0 : 1;
}
