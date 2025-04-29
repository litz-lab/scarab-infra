source /opt/ros/$ROS_DISTRO/setup.sh
source $tmpdir/autoware/install/setup.bash
nohup /opt/ros/humble/bin/ros2 bag play $tmpdir/driving_log_replayer_data/yabloc/sample/input_bag --topics /sensing/camera/traffic_light/image_raw/compressed -p &
sleep 5
nohup /opt/ros/humble/bin/ros2 bag play $tmpdir/autoware/lane_change/input_bag --topics /filtered/objects -p &
sleep 5
# cd $tmpdir/autoware/ && /opt/ros/humble/bin/ros2 launch tier4_map_launch map.launch.xml pointcloud_map_path:=/tmp_home/autoware_map/nishishinjuku_autoware_map/pointcloud_map.pcd pointcloud_map_metadata_path:='""' lanelet2_map_path:=/tmp_home/autoware_map/nishishinjuku_autoware_map/lanelet2_map.osm map_projector_info_path:='""' pointcloud_map_loader_param_path:=src/universe/autoware.universe/map/autoware_map_loader/config/pointcloud_map_loader.param.yaml lanelet2_map_loader_param_path:=src/universe/autoware.universe/map/autoware_map_loader/config/lanelet2_map_loader.param.yaml map_tf_generator_param_path:=src/universe/autoware.universe/map/autoware_map_tf_generator/config/map_tf_generator.param.yaml map_projection_loader_param_path:=/tmp_home/autoware/src/universe/autoware.universe/map/autoware_map_projection_loader/config/map_projection_loader.param.yaml &
# sleep 10
# source $tmpdir/autoware/image_republisher/install/setup.bash && /opt/ros/humble/bin/ros2 run image_republisher image_republisher &
# /opt/ros/humble/bin/ros2 run tf2_ros static_transform_publisher 0.5 0.0 1.2 0 0 0 base_link camera_linkd &
# /opt/ros/humble/bin/ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link traffic_light_left_camera/camera_optical_link &
# /opt/ros/humble/bin/ros2 launch yabloc_pose_initializer yabloc_pose_initializer.launch.xml camera_pose_initializer_param_path:=/tmp_home/autoware/src/universe/autoware.universe/localization/yabloc/yabloc_pose_initializer/config/camera_pose_initializer.param.yaml model_path:=$tmpdir/autoware_data/yabloc_pose_initializer/saved_model/model_float32.pb &
# sleep 5
# /opt/ros/humble/bin/ros2 topic pub --once /localization/pose_estimator/yabloc/image_processing/undistorted/camera_info sensor_msgs/msg/CameraInfo '{header: {stamp: {sec: 501, nanosec: 149988798}, frame_id: "traffic_light_left_camera/camera_optical_link"}, height: 720, width: 1280, distortion_model: "plumb_bob", d: [0.0, 0.0, 0.0, 0.0, 0.0], k: [365.71429443359375, 0.0, 640.5, 0.0, 365.4822082519531, 360.5, 0.0, 0.0, 1.0], r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], p: [365.71429443359375, 0.0, 640.5, 0.0, 0.0, 365.4822082519531, 360.5, 0.0, 0.0, 0.0, 1.0, 0.0], binning_x: 0, binning_y: 0, roi: {x_offset: 0, y_offset: 0, height: 0, width: 0, do_rectify: false}}'
# sleep 5
# $tmpdir/autoware/my_yabloc_pose_call_pkg/build/my_yabloc_pose_call_pkg/call_yabloc_pose
