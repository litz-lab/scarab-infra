#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>

// rosbag2_transport
#include <rosbag2_transport/player.hpp>
#include <rosbag2_transport/play_options.hpp>
#include <rosbag2_storage/storage_options.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <autoware_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_planning_msgs/msg/path.hpp>

// Lanelet2
#include <lanelet2_io/Io.h>
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_projection/LocalCartesian.h>

// Autoware stuff
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <tier4_planning_msgs/msg/path_with_lane_id.hpp>
#include <autoware/behavior_velocity_planner_common/planner_data.hpp>

#include "autoware_test_utils/autoware_test_utils.hpp"
#include "autoware_test_utils/mock_data_parser.hpp"

// CrosswalkModule, with PlannerParam definition
#include "scene_crosswalk.hpp"

#include <autoware/behavior_velocity_planner_common/utilization/util.hpp>

using nav_msgs::msg::Odometry;
using autoware_perception_msgs::msg::PredictedObjects;
using autoware_planning_msgs::msg::Path;
using autoware::test_utils::get_absolute_path_to_config;

using tier4_planning_msgs::msg::PathWithLaneId;
using autoware::behavior_velocity_planner::CrosswalkModule;
using PlannerParam  = CrosswalkModule::PlannerParam;

namespace avp = autoware::behavior_velocity_planner;
using autoware::behavior_velocity_planner::PlannerData;

template<typename T>
T get_or_declare_parameter(
  rclcpp::Node & node, const std::string & param_name, const T & default_value)
{
  // Declare with default if not declared yet
  if (!node.has_parameter(param_name)) {
    node.declare_parameter<T>(param_name, default_value);
  }
  // Then retrieve its value
  return node.get_parameter(param_name).get_value<T>();
}

/********************************************************************************
* CrosswalkTestNode
*******************************************************************************/
class CrosswalkTestNode : public rclcpp::Node
{
public:
  explicit CrosswalkTestNode(const std::string & map_path)
  : Node("crosswalk_test_node")
  {
    RCLCPP_INFO(this->get_logger(), "CrosswalkTestNode constructor...");

    lanelet::ErrorMessages errs;
    lanelet::Origin origin({0.0, 0.0});
    lanelet::projection::LocalCartesianProjector projector(origin);

    // Load
    auto map_unique = lanelet::load(map_path, projector, &errs);
    lanelet_map_ = lanelet::LaneletMapPtr(map_unique.release());
    for (const auto & e : errs) {
      RCLCPP_WARN(this->get_logger(), "Lanelet2 loading: %s", e.c_str());
    }

    const std::string ns = "crosswalk";

    PlannerParam param;
    // common
    param.show_processing_time =
      get_or_declare_parameter<bool>(*this, ns + ".common.show_processing_time", false);
    param.traffic_light_state_timeout =
      get_or_declare_parameter<double>(*this, ns + ".common.traffic_light_state_timeout", 3.0);

    // stop_position
    param.stop_position_threshold =
      get_or_declare_parameter<double>(*this, ns + ".stop_position.stop_position_threshold", 1.0);
    param.stop_distance_from_crosswalk =
      get_or_declare_parameter<double>(*this, ns + ".stop_position.stop_distance_from_crosswalk", 3.5);
    param.stop_distance_from_object_preferred =
      get_or_declare_parameter<double>(*this, ns + ".stop_position.stop_distance_from_object_preferred", 3.0);

    param.min_acc_preferred =
      get_or_declare_parameter<double>(*this, ns + ".stop_position.min_acc_preferred", -1.0);
    param.min_jerk_preferred =
      get_or_declare_parameter<double>(*this, ns + ".stop_position.min_jerk_preferred", -1.0);

    // restart_suppression
    param.min_dist_to_stop_for_restart_suppression =
      get_or_declare_parameter<double>(*this, ns + ".restart_suppression.min_distance_to_stop", 0.5);
    param.max_dist_to_stop_for_restart_suppression =
      get_or_declare_parameter<double>(*this, ns + ".restart_suppression.max_distance_to_stop", 1.0);

    // slow_down
    param.min_slow_down_velocity =
      get_or_declare_parameter<double>(*this, ns + ".slow_down.min_slow_down_velocity", 2.78);
    param.max_slow_down_jerk =
      get_or_declare_parameter<double>(*this, ns + ".slow_down.max_slow_down_jerk", -1.5);
    param.max_slow_down_accel =
      get_or_declare_parameter<double>(*this, ns + ".slow_down.max_slow_down_accel", -2.5);
    param.no_relax_velocity =
      get_or_declare_parameter<double>(*this, ns + ".slow_down.no_relax_velocity", 2.78);

    // stuck_vehicle
    param.enable_stuck_check_in_intersection =
      get_or_declare_parameter<bool>(*this, ns + ".stuck_vehicle.enable_stuck_check_in_intersection", false);
    param.stuck_vehicle_velocity =
      get_or_declare_parameter<double>(*this, ns + ".stuck_vehicle.stuck_vehicle_velocity", 1.0);
    param.max_stuck_vehicle_lateral_offset =
      get_or_declare_parameter<double>(*this, ns + ".stuck_vehicle.max_stuck_vehicle_lateral_offset", 2.0);
    param.required_clearance =
      get_or_declare_parameter<double>(*this, ns + ".stuck_vehicle.required_clearance", 6.0);
    param.min_acc_for_stuck_vehicle =
      get_or_declare_parameter<double>(*this, ns + ".stuck_vehicle.min_acc", -1.0);
    param.min_jerk_for_stuck_vehicle =
      get_or_declare_parameter<double>(*this, ns + ".stuck_vehicle.min_jerk", -1.0);
    param.max_jerk_for_stuck_vehicle =
      get_or_declare_parameter<double>(*this, ns + ".stuck_vehicle.max_jerk", 1.0);

    // pass_judge
    param.ego_pass_first_margin_x =
      get_or_declare_parameter<std::vector<double>>(*this, ns + ".pass_judge.ego_pass_first_margin_x", {0.0});
    param.ego_pass_first_margin_y =
      get_or_declare_parameter<std::vector<double>>(*this, ns + ".pass_judge.ego_pass_first_margin_y", {4.0});
    param.ego_pass_first_additional_margin =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.ego_pass_first_additional_margin", 0.5);
    param.ego_pass_later_margin_x =
      get_or_declare_parameter<std::vector<double>>(*this, ns + ".pass_judge.ego_pass_later_margin_x", {0.0});
    param.ego_pass_later_margin_y =
      get_or_declare_parameter<std::vector<double>>(*this, ns + ".pass_judge.ego_pass_later_margin_y", {13.0});
    param.ego_pass_later_additional_margin =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.ego_pass_later_additional_margin", 0.5);
    param.ego_min_assumed_speed =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.ego_min_assumed_speed", 2.0);

    param.min_acc_for_no_stop_decision =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.no_stop_decision.min_acc", -1.5);
    param.min_jerk_for_no_stop_decision =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.no_stop_decision.min_jerk", -1.5);

    param.stop_object_velocity =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.stop_object_velocity_threshold", 0.25);
    param.min_object_velocity =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.min_object_velocity", 1.39);
    param.disable_yield_for_new_stopped_object =
      get_or_declare_parameter<bool>(*this, ns + ".pass_judge.disable_yield_for_new_stopped_object", true);
    param.distance_set_for_no_intention_to_walk =
      get_or_declare_parameter<std::vector<double>>(*this, ns + ".pass_judge.distance_set_for_no_intention_to_walk", {1.0,5.0});
    param.timeout_set_for_no_intention_to_walk =
      get_or_declare_parameter<std::vector<double>>(*this, ns + ".pass_judge.timeout_set_for_no_intention_to_walk", {1.0,0.0});
    param.timeout_ego_stop_for_yield =
      get_or_declare_parameter<double>(*this, ns + ".pass_judge.timeout_ego_stop_for_yield", 1.0);

    // object_filtering
    param.crosswalk_attention_range =
      get_or_declare_parameter<double>(*this, ns + ".object_filtering.crosswalk_attention_range", 2.0);
    param.vehicle_object_cross_angle_threshold =
      get_or_declare_parameter<double>(*this, ns + ".object_filtering.vehicle_object_cross_angle_threshold", 0.5);
    param.look_unknown =
      get_or_declare_parameter<bool>(*this, ns + ".object_filtering.target_object.unknown", true);
    param.look_bicycle =
      get_or_declare_parameter<bool>(*this, ns + ".object_filtering.target_object.bicycle", true);
    param.look_motorcycle =
      get_or_declare_parameter<bool>(*this, ns + ".object_filtering.target_object.motorcycle", true);
    param.look_pedestrian =
      get_or_declare_parameter<bool>(*this, ns + ".object_filtering.target_object.pedestrian", true);

    // occlusion
    param.occlusion_enable =
      get_or_declare_parameter<bool>(*this, ns + ".occlusion.enable", true);
    param.occlusion_occluded_object_velocity =
      get_or_declare_parameter<double>(*this, ns + ".occlusion.occluded_object_velocity", 1.0);
    param.occlusion_slow_down_velocity =
      get_or_declare_parameter<double>(*this, ns + ".occlusion.slow_down_velocity", 1.0);
    param.occlusion_time_buffer =
      get_or_declare_parameter<double>(*this, ns + ".occlusion.time_buffer", 0.5);
    param.occlusion_min_size =
      get_or_declare_parameter<double>(*this, ns + ".occlusion.min_size", 1.0);
    param.occlusion_free_space_max =
      get_or_declare_parameter<int>(*this, ns + ".occlusion.free_space_max", 43);
    param.occlusion_occupied_min =
      get_or_declare_parameter<int>(*this, ns + ".occlusion.occupied_min", 58);
    param.occlusion_ignore_with_traffic_light =
      get_or_declare_parameter<bool>(*this, ns + ".occlusion.ignore_with_traffic_light", true);
    param.occlusion_ignore_behind_predicted_objects =
      get_or_declare_parameter<bool>(*this, ns + ".occlusion.ignore_behind_predicted_objects", true);

    param.occlusion_ignore_velocity_thresholds =
      get_or_declare_parameter<std::vector<double>>(
        *this, ns + ".occlusion.ignore_velocity_thresholds.custom_thresholds", {0.0});

    param.occlusion_extra_objects_size =
      get_or_declare_parameter<double>(*this, ns + ".occlusion.extra_predicted_objects_size", 0.5);

    crosswalk_id_ = 166;
    ego_lane_id_  = 49;

    crosswalk_module_ = std::make_shared<CrosswalkModule>(
      *this, ego_lane_id_, crosswalk_id_, std::nullopt,
      lanelet_map_, param, get_logger(), get_clock(),
      nullptr, nullptr);

    // debugging polygon
    // if (lanelet_map_->laneletLayer.exists(crosswalk_id_)) {
    //   auto llt = lanelet_map_->laneletLayer.get(crosswalk_id_);
    //   auto poly2d = llt.polygon2d();
    //   for (const auto & p : poly2d) {
    //     RCLCPP_INFO(this->get_logger(), "crosswalk vertex (%.3f, %.3f)", p.x(), p.y());
    //   }
    // }

    odom_sub_ = this->create_subscription<Odometry>(
      "/localization/kinematic_state", 10,
      [this](Odometry::SharedPtr msg) {
        odom_ = msg;
        tryRunModule();
      }
    );
    preds_sub_ = this->create_subscription<PredictedObjects>(
      "/perception/object_recognition/objects", 10,
      [this](PredictedObjects::SharedPtr msg) {
        preds_ = msg;
        tryRunModule();
      }
    );
    path_sub_ = this->create_subscription<Path>(
      "/planning/scenario_planning/lane_driving/behavior_planning/path", 10,
      [this](Path::SharedPtr msg) {
        path_ = msg;
        tryRunModule();
      }
    );
  }

private:
  [[nodiscard]] rclcpp::NodeOptions get_node_options() const
  {
    rclcpp::NodeOptions node_options;

    const auto common_param =
      get_absolute_path_to_config("autoware_test_utils", "test_common.param.yaml");
    const auto nearest_search_param =
      get_absolute_path_to_config("autoware_test_utils", "test_nearest_search.param.yaml");
    const auto vehicle_info_param =
      get_absolute_path_to_config("autoware_test_utils", "test_vehicle_info.param.yaml");

    std::string bpp_dir{"autoware_behavior_path_planner"};
    const auto bpp_param = get_absolute_path_to_config(bpp_dir, "behavior_path_planner.param.yaml");
    const auto drivable_area_expansion_param =
      get_absolute_path_to_config(bpp_dir, "drivable_area_expansion.param.yaml");
    const auto scene_module_manager_param =
      get_absolute_path_to_config(bpp_dir, "scene_module_manager.param.yaml");

    autoware::test_utils::updateNodeOptions(
      node_options,
      {common_param, nearest_search_param, vehicle_info_param,
      bpp_param, drivable_area_expansion_param,
      scene_module_manager_param});

    return node_options;
  }

  void tryRunModule()
  {
    if (odom_ && preds_ && path_) {
      RCLCPP_INFO(this->get_logger(), "All data ready => modifyPathVelocity start!");

      rclcpp::NodeOptions pd_opts;
      pd_opts.start_parameter_services(false)
             .automatically_declare_parameters_from_overrides(false)
             .parameter_overrides({
               {"max_accel",           rclcpp::ParameterValue(5.0)},
               {"max_jerk",            rclcpp::ParameterValue(5.0)},
               {"system_delay",        rclcpp::ParameterValue(0.3)},
               {"delay_response_time", rclcpp::ParameterValue(0.1)},
               // vehicle_info_utils
               {"wheel_radius",        rclcpp::ParameterValue(0.39)},
               {"wheel_width",         rclcpp::ParameterValue(0.42)},
               {"wheel_base",          rclcpp::ParameterValue(2.74)},
               {"wheel_tread",         rclcpp::ParameterValue(1.63)},
               {"front_overhang",      rclcpp::ParameterValue(1.0)},
               {"rear_overhang",       rclcpp::ParameterValue(1.03)},
               {"left_overhang",       rclcpp::ParameterValue(0.1)},
               {"right_overhang",      rclcpp::ParameterValue(0.1)},
               {"vehicle_height",      rclcpp::ParameterValue(2.5)},
               {"max_steer_angle",     rclcpp::ParameterValue(0.70)}
             });
      
      auto pd_node      = std::make_shared<rclcpp::Node>("planner_data_node", pd_opts);
      auto planner_data = std::make_shared<PlannerData>(*pd_node);

      // odom
      geometry_msgs::msg::PoseStamped ps;
      ps.header = odom_->header;
      ps.pose   = odom_->pose.pose;
      planner_data->current_odometry = std::make_shared<geometry_msgs::msg::PoseStamped>(ps);

      geometry_msgs::msg::TwistStamped ts;
      ts.header = odom_->header;
      ts.twist = odom_->twist.twist;
      planner_data->current_velocity = std::make_shared<geometry_msgs::msg::TwistStamped>(ts);

      // objects
      planner_data->predicted_objects = preds_;

      // path
      PathWithLaneId path_wlid;
      path_wlid.header = path_->header;
      for (const auto & pt : path_->points) {
        tier4_planning_msgs::msg::PathPointWithLaneId pp;
        pp.point.pose = pt.pose;
        pp.point.longitudinal_velocity_mps = 0.0f;
        pp.lane_ids.push_back(ego_lane_id_);
        path_wlid.points.push_back(pp);
      }

      {
        nav_msgs::msg::OccupancyGrid fake_grid;
        fake_grid.header.frame_id = "map";
        fake_grid.info.resolution = 0.5;   // 1 cell = 0.5 meter
        fake_grid.info.width = 100;       // 100 cells in x-direction
        fake_grid.info.height = 100;      // 100 cells in y-direction
        fake_grid.info.origin.position.x = 0.0;
        fake_grid.info.origin.position.y = 0.0;
        fake_grid.info.origin.orientation.w = 1.0;
      
        fake_grid.data.resize(fake_grid.info.width * fake_grid.info.height, 0);
      
        planner_data->occupancy_grid = std::make_shared<nav_msgs::msg::OccupancyGrid>(fake_grid);
      }

      using autoware::route_handler::RouteHandler;
      auto map_bin_msg = autoware::test_utils::make_map_bin_msg(
        "/tmp_home/autoware/crosswalk_velocity/crosswalk_map/crosswalk_map.osm", 5.0);
      
      auto route_handler_ptr = std::make_shared<RouteHandler>(map_bin_msg);
      
      planner_data->route_handler_ = route_handler_ptr;

      crosswalk_module_->setPlannerData(planner_data);
      bool result = crosswalk_module_->modifyPathVelocity(&path_wlid);

      RCLCPP_INFO(get_logger(), "modifyPathVelocity result: %s", result ? "true" : "false");

      odom_.reset();
      preds_.reset();
      path_.reset();
    }
  }

  lanelet::LaneletMapPtr lanelet_map_;
  std::shared_ptr<CrosswalkModule> crosswalk_module_;
  int crosswalk_id_;
  int ego_lane_id_;

  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<PredictedObjects>::SharedPtr preds_sub_;
  rclcpp::Subscription<Path>::SharedPtr path_sub_;

  Odometry::SharedPtr         odom_;
  PredictedObjects::SharedPtr preds_;
  Path::SharedPtr             path_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // rosbag2
  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri        = "/tmp_home/autoware/crosswalk_velocity/crosswalk_bag/crosswalk_bag.db3";
  storage_options.storage_id = "sqlite3";

  rosbag2_transport::PlayOptions play_options;
  play_options.read_ahead_queue_size = 1000;
  play_options.start_paused = true;
  play_options.rate = 1.0;

  auto player = std::make_shared<rosbag2_transport::Player>(
    storage_options, play_options, "my_bag_player_node");

  auto crosswalk_node = std::make_shared<CrosswalkTestNode>(
    "/tmp_home/autoware/crosswalk_velocity/crosswalk_map/crosswalk_map.osm");

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(player);
  exec.add_node(crosswalk_node);

  std::thread spin_thread([&exec]() {
    exec.spin();
  });

  // play_next()
  std::this_thread::sleep_for(std::chrono::seconds(1));
  for (int i = 1; i <= 3; i++) {
    RCLCPP_INFO(rclcpp::get_logger("main"), ">>> play_next() call #%d", i);
    player->play_next();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));
  RCLCPP_INFO(rclcpp::get_logger("main"), "Shutting down...");
  exec.cancel();
  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
