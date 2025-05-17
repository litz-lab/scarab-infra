#include <rclcpp/rclcpp.hpp>
#include <rosbag2_interfaces/srv/play_next.hpp>
#include <tier4_localization_msgs/srv/pose_with_covariance_stamped.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <cv_bridge/cv_bridge.h>

#include "yabloc_pose_initializer/camera/semantic_segmentation.hpp"

#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/ocl.hpp>

using ImageMsg = sensor_msgs::msg::CompressedImage;

namespace yabloc
{
class MySegmentationNode : public rclcpp::Node
{
public:
  explicit MySegmentationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("my_segmentation_node", options)
  {
    const std::string model_path = "/tmp_home/autoware_data/yabloc_pose_initializer/saved_model/model_float32.pb";
    RCLCPP_INFO_STREAM(get_logger(), "[MySegmentationNode] Using model path: " << model_path);

    if (std::filesystem::exists(model_path)) {
      semantic_segmentation_ = std::make_unique<SemanticSegmentation>(model_path);
      RCLCPP_INFO(get_logger(), "SemanticSegmentation model loaded successfully.");
    } else {
      RCLCPP_ERROR_STREAM(get_logger(), "Model file not found: " << model_path);
      return;
    }

    sub_image_ = create_subscription<ImageMsg>(
      "/sensing/camera/traffic_light/image_raw/compressed",
      rclcpp::SensorDataQoS(),
      std::bind(&MySegmentationNode::onImage, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "[MySegmentationNode] Node is ready. Waiting for images...");
  }

private:
  void onImage(const ImageMsg::SharedPtr msg)
  {
    if (!semantic_segmentation_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "No segmentation model loaded. Skip inference.");
      return;
    }

    cv::Mat src_image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (src_image.empty()) {
      RCLCPP_ERROR(get_logger(), "imdecode failed");  return;
    }

    cv::Mat segmented_image = semantic_segmentation_->inference(src_image);

    RCLCPP_INFO(
      get_logger(),
      "Segmentation done! Input: (%dx%d) => Output: (%dx%d)",
      src_image.cols, src_image.rows,
      segmented_image.cols, segmented_image.rows
    );

    rclcpp::shutdown();
  }

  std::unique_ptr<SemanticSegmentation> semantic_segmentation_;
  rclcpp::Subscription<ImageMsg>::SharedPtr sub_image_;
};
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  cv::ocl::setUseOpenCL(false);
  cv::setNumThreads(1);

  auto seg_node  = std::make_shared<yabloc::MySegmentationNode>();

  auto play_node = rclcpp::Node::make_shared("combined_client_node");

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(seg_node);
  exec.add_node(play_node);

  auto client = play_node->create_client<rosbag2_interfaces::srv::PlayNext>(
                  "/rosbag2_player/play_next");
  client->wait_for_service();
  client->async_send_request(std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>());

  exec.spin();
  rclcpp::shutdown();
}