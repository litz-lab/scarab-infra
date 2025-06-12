cmake_minimum_required(VERSION 3.8)
project(control_validator_full_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_interfaces REQUIRED)
find_package(autoware_control_msgs REQUIRED)
find_package(autoware_planning_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(autoware_control_validator REQUIRED)
find_package(autoware_internal_debug_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(autoware_universe_utils REQUIRED)
find_package(autoware_vehicle_info_utils REQUIRED)
find_package(diagnostic_updater REQUIRED)
find_package(autoware_signal_processing REQUIRED)

add_executable(control_validator_full_test src/control_validator_full_test.cpp)

target_include_directories(control_validator_full_test
  PRIVATE
    "/tmp_home/autoware/install/autoware_control_validator/include"
)

get_filename_component(ACV_CMAKE_DIR      "${autoware_control_validator_DIR}" DIRECTORY)
get_filename_component(ACV_PACKAGE_DIR    "${ACV_CMAKE_DIR}" DIRECTORY)
get_filename_component(ACV_INSTALL_DIR    "${ACV_PACKAGE_DIR}" DIRECTORY)

find_library(ACV_COMPONENT_LIB
  NAMES autoware_control_validator_component
  PATHS "${ACV_INSTALL_DIR}/lib"
  NO_DEFAULT_PATH
)

if(NOT ACV_COMPONENT_LIB)
  message(FATAL_ERROR "libautoware_control_validator_component.so")
endif()

target_link_libraries(control_validator_full_test
  "${ACV_COMPONENT_LIB}"
)

ament_target_dependencies(control_validator_full_test
  rclcpp
  rosbag2_interfaces
  autoware_control_msgs
  autoware_planning_msgs
  nav_msgs
  geometry_msgs
  autoware_control_validator
  autoware_internal_debug_msgs
  visualization_msgs
  autoware_universe_utils
  autoware_vehicle_info_utils
  diagnostic_updater
  autoware_signal_processing
)

install(TARGETS control_validator_full_test
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
