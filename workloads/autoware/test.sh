/opt/ros/humble/bin/ros2 bag play $tmpdir/driving_log_replayer_output/yabloc/latest/sample/result_bag --rate 0.5 --clock 200 --topics /localization/kinematic_state -p & BAG_PID=$!
ros2 service call /rosbag2_player/play_next rosbag2_interfaces/srv/PlayNext "{}"
kill $BAG_PID