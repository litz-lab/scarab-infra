source /opt/ros/$ROS_DISTRO/setup.sh
nohup /opt/ros/$ROS_DISTRO/bin/ros2 bag play \
  "$tmpdir/driving_log_replayer_output/yabloc/latest/sample/result_bag" \
  --rate 0.5 --clock 200 \
  --topics /localization/kinematic_state -p &
