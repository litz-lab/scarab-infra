cmake_minimum_required(VERSION 3.8)
project(euclidean_cluster)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosbag2_interfaces REQUIRED)
find_package(autoware_euclidean_cluster REQUIRED)
find_package(autoware_point_types REQUIRED)
find_package(tier4_perception_msgs REQUIRED)
find_package(rosbag2_transport   REQUIRED)

add_executable(cluster_test_exec src/cluster_test_node.cpp)
ament_target_dependencies(cluster_test_exec
  rclcpp sensor_msgs 
  rosbag2_interfaces
  autoware_euclidean_cluster 
  autoware_point_types 
  tier4_perception_msgs
  rosbag2_transport)

install(TARGETS cluster_test_exec
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
