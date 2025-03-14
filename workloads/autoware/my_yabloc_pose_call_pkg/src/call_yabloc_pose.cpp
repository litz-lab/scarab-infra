#include <rclcpp/rclcpp.hpp>
#include <rosbag2_interfaces/srv/play_next.hpp>
#include <tier4_localization_msgs/srv/pose_with_covariance_stamped.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("combined_client_node");

  {
    auto client_play_next = node->create_client<rosbag2_interfaces::srv::PlayNext>(
      "/rosbag2_player/play_next");

    while (!client_play_next->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_INFO(node->get_logger(), 
                  "Waiting for service /rosbag2_player/play_next...");
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(node->get_logger(), "Interrupted while waiting for the service.");
        return 1;
      }
    }

    auto request = std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>();
    auto future = client_play_next->async_send_request(request);

    RCLCPP_INFO(node->get_logger(), "Calling /rosbag2_player/play_next ...");
    auto status = rclcpp::spin_until_future_complete(node, future);
    if (status == rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_INFO(node->get_logger(), "Successfully called /rosbag2_player/play_next.");
    } else {
      RCLCPP_ERROR(node->get_logger(), "Failed to call /rosbag2_player/play_next.");
      rclcpp::shutdown();
      return 1;
    }
  }

  {
    using YablocSrv = tier4_localization_msgs::srv::PoseWithCovarianceStamped;
    auto client_yabloc = node->create_client<YablocSrv>(
      "/localization/pose_estimator/yabloc/initializer/yabloc_align_srv");

    while (!client_yabloc->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_INFO(node->get_logger(), 
                  "Waiting for /localization/pose_estimator/yabloc/initializer/yabloc_align_srv...");
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(node->get_logger(), "Interrupted while waiting for the service.");
        return 1;
      }
    }

    auto request = std::make_shared<YablocSrv::Request>();
    request->pose_with_covariance.header.stamp.sec = 0;
    request->pose_with_covariance.header.stamp.nanosec = 0;
    request->pose_with_covariance.header.frame_id = "map";
    request->pose_with_covariance.pose.pose.position.x = 100.0;
    request->pose_with_covariance.pose.pose.position.y = 50.0;
    request->pose_with_covariance.pose.pose.position.z = 0.0;
    request->pose_with_covariance.pose.pose.orientation.x = 0.0;
    request->pose_with_covariance.pose.pose.orientation.y = 0.0;
    request->pose_with_covariance.pose.pose.orientation.z = 0.0;
    request->pose_with_covariance.pose.pose.orientation.w = 1.0;
    for (size_t i = 0; i < 36; i++) {
      request->pose_with_covariance.pose.covariance[i] = 0.0;
    }

    RCLCPP_INFO(node->get_logger(), "Calling /localization/pose_estimator/yabloc/initializer/yabloc_align_srv ...");
    auto future = client_yabloc->async_send_request(request);

    auto status = rclcpp::spin_until_future_complete(node, future);
    if (status == rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_INFO(node->get_logger(), "Successfully called yabloc_align_srv.");
    } else {
      RCLCPP_ERROR(node->get_logger(), "Failed to call yabloc_align_srv.");
    }
  }

  rclcpp::shutdown();
  return 0;
}
