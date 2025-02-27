# syntax = edrevo/dockerfile-plus

FROM ghcr.io/autowarefoundation/autoware:universe-devel-0.39.1-amd64
SHELL ["/bin/bash", "-c"]

INCLUDE+ ./common/Dockerfile.common

WORKDIR /autoware
RUN mkdir -p result_bag
COPY ./workloads/autoware/result_bag/ /autoware/result_bag/

# RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
#     ros2 bag play /autoware/result_bag/ --rate 0.5 --clock 200
 
# Start your application
CMD ["/bin/bash"]
