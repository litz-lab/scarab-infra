cmake_minimum_required(VERSION 3.8)
project(mpc_controller_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(autoware_control_msgs REQUIRED)
find_package(autoware_planning_msgs REQUIRED)
find_package(autoware_vehicle_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(autoware_mpc_lateral_controller REQUIRED)
find_package(rosbag2_interfaces REQUIRED)

add_executable(mpc_controller_exec src/test_mpc_controller_test.cpp)
ament_target_dependencies(mpc_controller_exec
  rclcpp
  autoware_control_msgs
  autoware_planning_msgs
  autoware_vehicle_msgs
  nav_msgs
  autoware_mpc_lateral_controller
  rosbag2_interfaces
)

install(TARGETS mpc_controller_exec
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
