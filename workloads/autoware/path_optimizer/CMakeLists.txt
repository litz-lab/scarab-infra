cmake_minimum_required(VERSION 3.8)
project(path_optimizer)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(autoware_test_utils REQUIRED)
find_package(autoware_planning_test_manager REQUIRED)
find_package(tier4_planning_msgs REQUIRED)
find_package(autoware_motion_utils REQUIRED)
find_package(autoware_path_optimizer REQUIRED)
find_package(rosbag2_interfaces REQUIRED)

add_executable(path_optimizer_exec src/path_optimizer.cpp)
ament_target_dependencies(path_optimizer_exec
  rclcpp
  autoware_test_utils
  autoware_planning_test_manager
  tier4_planning_msgs
  autoware_motion_utils
  autoware_path_optimizer
  rosbag2_interfaces
)
install(TARGETS path_optimizer_exec
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
