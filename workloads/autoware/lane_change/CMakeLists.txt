cmake_minimum_required(VERSION 3.8)
project(lane_change)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(GTest REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(autoware_behavior_path_lane_change_module REQUIRED)
find_package(autoware_behavior_path_planner REQUIRED)
find_package(autoware_behavior_path_planner_common REQUIRED)
find_package(autoware_frenet_planner REQUIRED)
find_package(autoware_motion_utils REQUIRED)
find_package(autoware_rtc_interface REQUIRED)
find_package(autoware_universe_utils REQUIRED)
find_package(fmt REQUIRED)
find_package(pluginlib REQUIRED)
find_package(range-v3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tier4_planning_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosbag2_interfaces REQUIRED)

add_executable(lane_change_exec src/my_lane_change_node.cpp)
ament_target_dependencies(lane_change_exec
  autoware_behavior_path_lane_change_module
  autoware_behavior_path_planner
  autoware_behavior_path_planner_common
  autoware_frenet_planner
  autoware_motion_utils
  autoware_rtc_interface
  autoware_universe_utils
  fmt
  pluginlib
  range-v3
  rclcpp
  tier4_planning_msgs
  visualization_msgs
  rosbag2_interfaces
)

target_link_libraries(lane_change_exec
  GTest::gtest
  GTest::gtest_main
)

install(TARGETS lane_change_exec
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
