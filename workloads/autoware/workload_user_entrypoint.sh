source /opt/ros/$ROS_DISTRO/setup.sh
source $tmpdir/autoware/install/setup.bash
nohup /opt/ros/humble/bin/ros2 bag play $tmpdir/driving_log_replayer_data/yabloc/sample/input_bag --topics /sensing/camera/traffic_light/image_raw/compressed -p &
sleep 5
nohup /opt/ros/humble/bin/ros2 bag play $tmpdir/autoware/lane_change/input_bag --topics /filtered/objects -p &
sleep 5
