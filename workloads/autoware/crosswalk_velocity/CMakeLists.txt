cmake_minimum_required(VERSION 3.8)
project(crosswalk_velocity)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(lanelet2_core REQUIRED)
find_package(lanelet2_io REQUIRED)
find_package(lanelet2_projection REQUIRED)
find_package(autoware_lanelet2_extension REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tier4_planning_msgs REQUIRED)
find_package(autoware_behavior_velocity_crosswalk_module REQUIRED)
find_package(rosbag2_transport REQUIRED)

add_executable(crosswalk_test_exec src/crosswalk_test.cpp)
ament_target_dependencies(crosswalk_test_exec
  rclcpp
  rosbag2_cpp
  lanelet2_core
  lanelet2_io
  lanelet2_projection
  autoware_lanelet2_extension 
  nav_msgs
  tier4_planning_msgs
  autoware_behavior_velocity_crosswalk_module
  rosbag2_transport
)

install(TARGETS crosswalk_test_exec
  DESTINATION lib/${PROJECT_NAME}
)

target_include_directories(crosswalk_test_exec
  PRIVATE
    /tmp_home/autoware/src/universe/autoware.universe/planning/behavior_velocity_planner/autoware_behavior_velocity_crosswalk_module/src
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
