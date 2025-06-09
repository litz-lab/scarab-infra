#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "rosbag2_interfaces/srv/play_next.hpp"
#include "cv_bridge/cv_bridge.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/ocl.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/core/ocl.hpp>

namespace my_ros2_module
{

class LaneDetectorCpp : public rclcpp::Node
{
public:
  using ImageMsg = sensor_msgs::msg::CompressedImage;

  std::atomic_bool image_processed_;
  explicit LaneDetectorCpp(const rclcpp::NodeOptions & options)
  : Node("lane_detector_cpp", options),
    image_processed_(false)
  {
    using namespace std::placeholders;

    sub_image_ = create_subscription<ImageMsg>("/sensing/camera/traffic_light/image_raw/compressed",
      rclcpp::SensorDataQoS(),
      std::bind(&LaneDetectorCpp::imageCallback, this, std::placeholders::_1));

    pub_image_ = create_publisher<sensor_msgs::msg::Image>("/custom/line_detection_result_final", 10);

    RCLCPP_INFO(get_logger(), "LaneDetectorCpp node has started.");
  }

private:
  void imageCallback(const ImageMsg::SharedPtr msg)
  {
    cv::Mat bgr_image =
      cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (bgr_image.empty()) {
      RCLCPP_ERROR(get_logger(), "imdecode failed");
      return;
    }

    cv::Mat gray_image;
    cv::cvtColor(bgr_image, gray_image, cv::COLOR_BGR2GRAY);

    int height = gray_image.rows;
    int width = gray_image.cols;
    cv::Mat roi_mask = cv::Mat::zeros(gray_image.size(), CV_8UC1);
    std::vector<cv::Point> polygon{
      cv::Point(static_cast<int>(0.1 * width), height),
      cv::Point(static_cast<int>(0.45 * width), static_cast<int>(0.6 * height)),
      cv::Point(static_cast<int>(0.55 * width), static_cast<int>(0.6 * height)),
      cv::Point(static_cast<int>(0.9 * width), height)
    };
    cv::fillPoly(roi_mask, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(255));
    cv::Mat roi_gray;
    cv::bitwise_and(gray_image, roi_mask, roi_gray);

    auto start = std::chrono::steady_clock::now();
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    cv::Mat lines;
    lsd->detect(roi_gray, lines);
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    RCLCPP_INFO(get_logger(), "LSD detection time: %.2f ms", elapsed_ms);

    cv::Mat final_img;
    cv::cvtColor(roi_gray, final_img, cv::COLOR_GRAY2BGR);
    if (!lines.empty()) {
      RCLCPP_INFO(get_logger(), "Number of lines detected: %d", lines.rows);
      lsd->drawSegments(final_img, lines);
    } else {
      RCLCPP_INFO(get_logger(), "No lines detected.");
    }

    cv_bridge::CvImage out_msg;
    out_msg.header = msg->header;
    out_msg.encoding = sensor_msgs::image_encodings::BGR8;
    out_msg.image = final_img;
    pub_image_->publish(*out_msg.toImageMsg());

    image_processed_ = true;
    rclcpp::shutdown(); 
  }

  rclcpp::Subscription<ImageMsg>::SharedPtr sub_image_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_;
};

}  // namespace my_ros2_module

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  cv::ocl::setUseOpenCL(false);
  cv::setNumThreads(1);

  auto node = std::make_shared<my_ros2_module::LaneDetectorCpp>(rclcpp::NodeOptions());

  auto play_node = rclcpp::Node::make_shared("combined_client_node");

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);
  exec.add_node(play_node);

  auto client = play_node->create_client<rosbag2_interfaces::srv::PlayNext>("/rosbag2_player/play_next");
  client->wait_for_service();
  client->async_send_request(std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>());

  exec.spin();
  rclcpp::shutdown();

  return 0;
}