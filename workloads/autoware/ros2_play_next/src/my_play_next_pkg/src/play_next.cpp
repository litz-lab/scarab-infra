#include <rclcpp/rclcpp.hpp>
#include <rosbag2_interfaces/srv/play_next.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("play_next_client");
  auto client = node->create_client<rosbag2_interfaces::srv::PlayNext>("/rosbag2_player/play_next");

  auto request = std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>();
  auto future = client->async_send_request(request);

  rclcpp::spin_until_future_complete(node, future);

  rclcpp::shutdown();
  return 0;
}
