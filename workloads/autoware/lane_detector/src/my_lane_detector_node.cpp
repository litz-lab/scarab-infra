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

#include <sensor_msgs/image_encodings.hpp>

namespace my_ros2_module
{

class LaneDetectorCpp : public rclcpp::Node
{
public:
  explicit LaneDetectorCpp(const rclcpp::NodeOptions & options)
  : Node("lane_detector_cpp", options),
    image_processed_(false)
  {
    using namespace std::placeholders;

    sub_image_ = create_subscription<sensor_msgs::msg::Image>(
      "/localization/pose_estimator/yabloc/image_processing/undistorted/image_raw", 10,
      std::bind(&LaneDetectorCpp::imageCallback, this, _1));

    pub_image_ = create_publisher<sensor_msgs::msg::Image>(
      "/custom/line_detection_result_final", 10);

    RCLCPP_INFO(get_logger(), "LaneDetectorCpp node has started.");
  }

private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (std::exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat bgr_image = cv_ptr->image;

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

    RCLCPP_INFO(get_logger(), "Published final processed image message.");

    image_processed_ = true;
    rclcpp::shutdown();
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_image_;
  std::atomic_bool image_processed_;
};

}  // namespace my_ros2_module

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<my_ros2_module::LaneDetectorCpp>(rclcpp::NodeOptions());

  auto client_play_next = node->create_client<rosbag2_interfaces::srv::PlayNext>("/rosbag2_player/play_next");
  
  while (!client_play_next->wait_for_service(std::chrono::seconds(1))) {
    RCLCPP_INFO(node->get_logger(), "Waiting for service /rosbag2_player/play_next...");
    if (!rclcpp::ok()) {
      RCLCPP_ERROR(node->get_logger(), "Interrupted while waiting for service.");
      rclcpp::shutdown();
      return 1;
    }
  }

  auto request = std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>();
  RCLCPP_INFO(node->get_logger(), "Calling /rosbag2_player/play_next...");
  auto future = client_play_next->async_send_request(request);
  auto status = rclcpp::spin_until_future_complete(node, future);
  if (status == rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_INFO(node->get_logger(), "Successfully called /rosbag2_player/play_next.");
  } else {
    RCLCPP_ERROR(node->get_logger(), "Failed to call /rosbag2_player/play_next.");
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  return 0;
}
