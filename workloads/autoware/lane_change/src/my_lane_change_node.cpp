#include "autoware/behavior_path_lane_change_module/scene.hpp"
#include "autoware/behavior_path_lane_change_module/manager.hpp"
#include "autoware/behavior_path_planner_common/data_manager.hpp"
#include "autoware_test_utils/autoware_test_utils.hpp"
#include "autoware_test_utils/mock_data_parser.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>

#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <limits>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>

#include <rosbag2_interfaces/srv/play_next.hpp>
#include <chrono>

using autoware::behavior_path_planner::LaneChangeModuleManager;
using autoware::behavior_path_planner::LaneChangeModuleType;
using autoware::behavior_path_planner::NormalLaneChange;
using autoware::behavior_path_planner::PlannerData;
using autoware::behavior_path_planner::lane_change::CommonDataPtr;
using autoware::behavior_path_planner::lane_change::LCParamPtr;
using autoware::behavior_path_planner::lane_change::RouteHandlerPtr;
using autoware::route_handler::Direction;
using autoware::route_handler::RouteHandler;
using autoware::test_utils::get_absolute_path_to_config;
using autoware::test_utils::get_absolute_path_to_lanelet_map;
using autoware::test_utils::get_absolute_path_to_route;

using autoware_map_msgs::msg::LaneletMapBin;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_planning_msgs::msg::LaneletRoute;
using geometry_msgs::msg::Pose;
using tier4_planning_msgs::msg::PathWithLaneId;

class NormalLaneChangeTest : public autoware::behavior_path_planner::NormalLaneChange
{
public:
  NormalLaneChangeTest(
    const autoware::behavior_path_planner::lane_change::LCParamPtr & param_ptr,
    const autoware::behavior_path_planner::LaneChangeModuleType & type,
    const autoware::route_handler::Direction & direction)
  : NormalLaneChange(param_ptr, type, direction) {}

  using NormalLaneChange::prev_module_output_;
  using NormalLaneChange::common_data_ptr_;
};

class TestNormalLaneChangeNoGTest
{
public:
  TestNormalLaneChangeNoGTest() = default;

  ~TestNormalLaneChangeNoGTest()
  {
    normal_lane_change_ = nullptr;
    lc_param_ptr_ = nullptr;
    planner_data_ = nullptr;
    rclcpp::shutdown();
  }

  void init_param()
  {
    rclcpp::init(0, nullptr);

    auto node_options = get_node_options();
    node_ = std::make_shared<rclcpp::Node>(test_node_name_, node_options);

    // Initialize planner data
    planner_data_->init_parameters(*node_);

    // Lane change parameters
    lc_param_ptr_ = LaneChangeModuleManager::set_params(node_.get(), node_->get_name());
    planner_data_->route_handler = init_route_handler();

    // Ego pose
    ego_pose_ = autoware::test_utils::createPose(-50.0, 1.75, 0.0, 0.0, 0.0, 0.0);
    planner_data_->self_odometry = set_odometry(ego_pose_);

    sub_objects_ = node_->create_subscription<PredictedObjects>(
      "/filtered/objects",
      rclcpp::QoS{1}.best_effort(),
      [this](const PredictedObjects::SharedPtr msg) {
        RCLCPP_INFO(node_->get_logger(), "Received PredictedObjects from topic. size=%zu", msg->objects.size());
        planner_data_->dynamic_object = msg;

        this->init_module();
        this->checkLaneChangeStatus();
    
        std::cout << "[INFO] All checks passed successfully.\n";
      }
    );
  }

  void init_module()
  {
    normal_lane_change_ = std::make_shared<NormalLaneChangeTest>(
      lc_param_ptr_, lc_type_, lc_direction_);

    // Provide it with the core data
    normal_lane_change_->setData(planner_data_);

    // Set an example "previous approved path"
    normal_lane_change_->prev_module_output_.path = create_previous_approved_path();
  }

  void checkLaneChangeStatus()
  {
    constexpr auto is_approved = true;

    // First, call update_lanes with is_approved == true
    normal_lane_change_->update_lanes(is_approved);

    // We expect "lanes_available" to be false
    if (normal_lane_change_->common_data_ptr_->is_lanes_available()) {
      std::cerr << "[ERROR] Lanes should NOT be available when is_approved == true.\n";
    } else {
      std::cout << "[PASS] Lanes are not available as expected (is_approved == true).\n";
    }

    // Second, call update_lanes with is_approved == false
    normal_lane_change_->update_lanes(!is_approved);

    // We expect "lanes_available" to be true
    if (!normal_lane_change_->common_data_ptr_->is_lanes_available()) {
      std::cerr << "[ERROR] Lanes should be available when is_approved == false.\n";
    } else {
      std::cout << "[PASS] Lanes are available as expected (is_approved == false).\n";
    }
  }

  bool hasObjects() const
  {
    return planner_data_->dynamic_object && !planner_data_->dynamic_object->objects.empty();
  }

  rclcpp::Node::SharedPtr getNode() const
  {
    return node_;
  }

private:
  [[nodiscard]] rclcpp::NodeOptions get_node_options() const
  {
    rclcpp::NodeOptions node_options;

    const auto common_param =
      get_absolute_path_to_config(test_utils_dir_, "test_common.param.yaml");
    const auto nearest_search_param =
      get_absolute_path_to_config(test_utils_dir_, "test_nearest_search.param.yaml");
    const auto vehicle_info_param =
      get_absolute_path_to_config(test_utils_dir_, "test_vehicle_info.param.yaml");

    std::string bpp_dir{"autoware_behavior_path_planner"};
    const auto bpp_param = get_absolute_path_to_config(bpp_dir, "behavior_path_planner.param.yaml");
    const auto drivable_area_expansion_param =
      get_absolute_path_to_config(bpp_dir, "drivable_area_expansion.param.yaml");
    const auto scene_module_manager_param =
      get_absolute_path_to_config(bpp_dir, "scene_module_manager.param.yaml");

    std::string lc_dir{"autoware_behavior_path_lane_change_module"};
    const auto lc_param = get_absolute_path_to_config(lc_dir, "lane_change.param.yaml");

    autoware::test_utils::updateNodeOptions(
      node_options,
      {common_param, nearest_search_param, vehicle_info_param,
       bpp_param, drivable_area_expansion_param,
       scene_module_manager_param, lc_param});

    return node_options;
  }

  [[nodiscard]] RouteHandlerPtr init_route_handler() const
  {
    std::string autoware_route_handler_dir{"autoware_route_handler"};
    std::string lane_change_right_test_route_filename{"lane_change_test_route.yaml"};
    std::string lanelet_map_filename{"2km_test.osm"};

    const auto lanelet2_path =
      get_absolute_path_to_lanelet_map(test_utils_dir_, lanelet_map_filename);

    // Load lanelet map into LaneletMapBin
    const auto map_bin_msg = autoware::test_utils::make_map_bin_msg(lanelet2_path, 5.0);

    auto route_handler_ptr = std::make_shared<RouteHandler>(map_bin_msg);

    // Load route
    const auto rh_test_route = get_absolute_path_to_route(
      autoware_route_handler_dir, lane_change_right_test_route_filename);

    if (const auto route_opt = autoware::test_utils::parse<std::optional<LaneletRoute>>(rh_test_route)) {
      route_handler_ptr->setRoute(*route_opt);
    }

    return route_handler_ptr;
  }

  [[nodiscard]] std::shared_ptr<nav_msgs::msg::Odometry> set_odometry(const Pose & pose) const
  {
    nav_msgs::msg::Odometry odom;
    odom.pose.pose = pose;
    return std::make_shared<nav_msgs::msg::Odometry>(odom);
  }

  [[nodiscard]] tier4_planning_msgs::msg::PathWithLaneId create_previous_approved_path() const
  {
    auto route_handler_ptr = planner_data_->route_handler;

    lanelet::ConstLanelet closest_lane;
    route_handler_ptr->getClosestLaneletWithinRoute(ego_pose_, &closest_lane);

    const double backward_distance = 30.0;
    const double forward_distance  = 100.0;

    const auto current_lanes = route_handler_ptr->getLaneletSequence(
      closest_lane, ego_pose_, backward_distance, forward_distance);

    return route_handler_ptr->getCenterLinePath(
      current_lanes, 0.0, std::numeric_limits<double>::max());
  }

private:
  rclcpp::Subscription<PredictedObjects>::SharedPtr sub_objects_;

  rclcpp::Node::SharedPtr node_;
  std::string test_node_name_{"test_lane_change_scene"};
  std::string test_utils_dir_{"autoware_test_utils"};

  geometry_msgs::msg::Pose ego_pose_;

  LCParamPtr lc_param_ptr_;
  std::shared_ptr<NormalLaneChangeTest> normal_lane_change_;
  std::shared_ptr<PlannerData> planner_data_ = std::make_shared<PlannerData>();

  LaneChangeModuleType lc_type_{LaneChangeModuleType::NORMAL};
  Direction lc_direction_{Direction::RIGHT};
};

int main(int, char **)
{
  try {
    TestNormalLaneChangeNoGTest test;

    test.init_param();

    auto node_handle = test.getNode();

    auto client_play_next =
      node_handle->create_client<rosbag2_interfaces::srv::PlayNext>("/rosbag2_player/play_next");

    while (!client_play_next->wait_for_service(std::chrono::seconds(1))) {
      RCLCPP_INFO(node_handle->get_logger(), "Waiting for service /rosbag2_player/play_next...");
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(node_handle->get_logger(), "Interrupted while waiting for service.");
        rclcpp::shutdown();
        return 1;
      }
    }

    auto request = std::make_shared<rosbag2_interfaces::srv::PlayNext::Request>();
    RCLCPP_INFO(node_handle->get_logger(), "Calling /rosbag2_player/play_next...");
    auto future = client_play_next->async_send_request(request);

    auto status = rclcpp::spin_until_future_complete(node_handle, future);
    if (status == rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_INFO(node_handle->get_logger(), "Successfully called /rosbag2_player/play_next.");
    } else {
      RCLCPP_ERROR(node_handle->get_logger(), "Failed to call /rosbag2_player/play_next.");
      rclcpp::shutdown();
      return 1;
    }
  } catch (const std::exception & e) {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
