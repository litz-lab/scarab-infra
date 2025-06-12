#include <memory>
#include <string>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tier4_perception_msgs/msg/detected_objects_with_feature.hpp"
#include "autoware/euclidean_cluster/voxel_grid_based_euclidean_cluster.hpp"

// Helper to set PointCloud2 fields
void setPointCloud2Fields(sensor_msgs::msg::PointCloud2 & pointcloud)
{
  pointcloud.fields.resize(4);
  pointcloud.fields[0].name = "x";
  pointcloud.fields[1].name = "y";
  pointcloud.fields[2].name = "z";
  pointcloud.fields[3].name = "intensity";
  pointcloud.fields[0].offset = 0;
  pointcloud.fields[1].offset = 4;
  pointcloud.fields[2].offset = 8;
  pointcloud.fields[3].offset = 12;
  pointcloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
  pointcloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
  pointcloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
  pointcloud.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
  pointcloud.fields[0].count = 1;
  pointcloud.fields[1].count = 1;
  pointcloud.fields[2].count = 1;
  pointcloud.fields[3].count = 1;
  pointcloud.height = 1;
  pointcloud.point_step = 16;
  pointcloud.is_bigendian = false;
  pointcloud.is_dense = true;
  pointcloud.header.frame_id = "dummy_frame_id";
  pointcloud.header.stamp.sec = 0;
  pointcloud.header.stamp.nanosec = 0;
}

// Function to generate a cluster with a large number of points
sensor_msgs::msg::PointCloud2 generateClusterWithLargePoints(const int nb_points)
{
  sensor_msgs::msg::PointCloud2 pointcloud;
  setPointCloud2Fields(pointcloud);
  pointcloud.data.resize(nb_points * pointcloud.point_step);

  // Generate one cluster with specified number of points
  pcl::PointXYZI point;

  // Random number generator for x, y, z coordinates (in a more confined range for clustering)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 3.0);  // Increased range to 0.0 ~ 3.0

  for (int i = 0; i < nb_points; ++i) {
    point.x = dis(gen);  // point.x within 0.0 to 3.0
    point.y = dis(gen);  // point.y within 0.0 to 3.0
    point.z = dis(gen);  // point.z within 0.0 to 3.0
    point.intensity = 0.0;
    memcpy(&pointcloud.data[i * pointcloud.point_step], &point, pointcloud.point_step);
  }
  pointcloud.width = nb_points;
  pointcloud.row_step = pointcloud.point_step * nb_points;
  return pointcloud;
}

namespace cluster_test
{
class ClusterTestNode : public rclcpp::Node
{
public:
  explicit ClusterTestNode(uint64_t start_stamp)
  : Node("cluster_test_node_offline"),
    start_stamp_(start_stamp)
  {
    const bool  use_height                  = false;
    const int   min_cluster_size            = 10;    // Increase min cluster size
    const int   max_cluster_size            = 196608;  // Increase max cluster size
    const float tolerance                   = 0.7f;  // Increased tolerance
    const float voxel_leaf_size             = 0.3f;  // Reduced voxel leaf size
    const int   min_points_number_per_voxel = 1;

    cluster_ = std::make_shared<
      autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster>(
        use_height, min_cluster_size, max_cluster_size,
        tolerance, voxel_leaf_size, min_points_number_per_voxel);

    RCLCPP_INFO(get_logger(), "Cluster test started with start timestamp: %lu", static_cast<unsigned long>(start_stamp_));
  }

  bool run()
  {
    // Generate a large cluster (test large points generation)
    int nb_generated_points = 196608;  // Larger number of points
    sensor_msgs::msg::PointCloud2 pointcloud = generateClusterWithLargePoints(nb_generated_points);

    const sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud_msg =
      std::make_shared<sensor_msgs::msg::PointCloud2>(pointcloud);

    tier4_perception_msgs::msg::DetectedObjectsWithFeature output;

    // Perform clustering
    if (cluster_->cluster(pointcloud_msg, output)) {
      std::cout << "Cluster success" << std::endl;
    } else {
      std::cout << "Cluster failed" << std::endl;
    }

    // Output the number of clusters
    std::cout << "Number of output clusters: " << output.feature_objects.size() << std::endl;
    if (!output.feature_objects.empty()) {
      std::cout << "Number of points in the first cluster: " << output.feature_objects[0].feature.cluster.width << std::endl;
    }

    // The output clusters should have only one cluster with nb_generated_points points
    if (output.feature_objects.size() == 1) {
      std::cout << "Test Passed: Found 1 cluster with " << output.feature_objects[0].feature.cluster.width << " points" << std::endl;
    } else {
      std::cout << "Test Failed: Unexpected number of clusters" << std::endl;
    }

    return true;
  }

private:
  uint64_t start_stamp_;
  std::shared_ptr<
    autoware::euclidean_cluster::VoxelGridBasedEuclideanCluster> cluster_;
};
}  // namespace cluster_test

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  uint64_t stamp = (argc > 1) ? std::stoull(argv[1]) : 0;

  auto node = std::make_shared<cluster_test::ClusterTestNode>(stamp);
  for (int i = 0; i < 1; i++)
    bool ok = node->run();
  RCLCPP_INFO(node->get_logger(), "Cluster test end with timestamp");
  rclcpp::shutdown();
  return 0;
}
