source /opt/ros/$ROS_DISTRO/setup.sh
source $tmpdir/autoware/install/setup.bash
nohup /opt/ros/humble/bin/ros2 bag play $tmpdir/driving_log_replayer_data/yabloc/sample/input_bag --topics /sensing/camera/traffic_light/image_raw/compressed -p &
