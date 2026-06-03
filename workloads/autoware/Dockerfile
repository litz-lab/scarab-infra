# syntax = edrevo/dockerfile-plus

FROM ghcr.io/autowarefoundation/autoware:universe-devel-0.39.1-amd64
SHELL ["/bin/bash", "-c"]

USER root
RUN rm -f /etc/apt/sources.list.d/ros2* /etc/apt/sources.list.d/ros-* 2>/dev/null || true && \
    (apt-get update || true) && \
    apt-get install -y --no-install-recommends curl gnupg ca-certificates && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get install -y --no-install-recommends gnupg lsb-release && \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
      | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-18 main" \
      > /etc/apt/sources.list.d/llvm18.list && \
    apt-get update


INCLUDE+ ./common/Dockerfile.common

USER root
# Switch to the temporary working directory
RUN mv /autoware $tmpdir/
WORKDIR $tmpdir

# TODO: Install required packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y pipx && \
    add-apt-repository universe && \
    add-apt-repository multiverse && \
    apt-get update && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-utils \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-vcstool \
      python3-pip \
      make \
      zstd \
      unzip \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN rosdep init || true  && rosdep fix-permissions && rosdep update

# Clone the autoware repos into a temporary folder
WORKDIR /tmp/autoware_repos
RUN git clone --depth 1 --filter=blob:none --sparse --branch 0.41.2 \
    https://github.com/autowarefoundation/autoware.git . && \
    git sparse-checkout set autoware.repos simulator.repos

# Move the .repos files into $tmpdir/autoware
RUN mkdir -p $tmpdir/autoware && \
    mv autoware.repos $tmpdir/autoware/ && \
    mv simulator.repos $tmpdir/autoware/ && \
    rm -rf /tmp/autoware_repos

# Copy simulator_append.txt to ros2 custom packates(play_next)
COPY ./workloads/autoware/simulator_append.txt $tmpdir/autoware/

WORKDIR $tmpdir/autoware
RUN cat simulator_append.txt >> simulator.repos && \
    mkdir -p src

# Autoware model
RUN cd $tmpdir/autoware && /root/.local/bin/ansible-galaxy collection install -f -r "ansible-galaxy-requirements.yaml" && \
    /root/.local/bin/ansible-playbook autoware.dev_env.download_artifacts -e "data_dir=$tmpdir/autoware_data"

# Import repos
RUN vcs import src < autoware.repos && \
    vcs import src < simulator.repos

# Build Autoware
RUN sed -i 's/^\(\s*\)private:/\1public:/' $tmpdir/autoware/src/universe/autoware.universe/control/autoware_control_validator/include/autoware/control_validator/control_validator.hpp && \
    source /opt/ros/$ROS_DISTRO/setup.bash && \
    apt-get update && \
    rosdep fix-permissions && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro $ROS_DISTRO && \
    source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --base-paths src --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

COPY ./workloads/autoware/my_yabloc_pose_call_pkg $tmpdir/autoware/my_yabloc_pose_call_pkg
COPY ./workloads/autoware/lane_detector $tmpdir/autoware/lane_detector
COPY ./workloads/autoware/lane_change $tmpdir/autoware/lane_change
COPY ./workloads/autoware/mpc_controller_test $tmpdir/autoware/mpc_controller_test
COPY ./workloads/autoware/euclidean_cluster $tmpdir/autoware/euclidean_cluster
COPY ./workloads/autoware/crosswalk_velocity $tmpdir/autoware/crosswalk_velocity
COPY ./workloads/autoware/pid_longitudinal_controller_test $tmpdir/autoware/pid_longitudinal_controller_test
COPY ./workloads/autoware/path_optimizer $tmpdir/autoware/path_optimizer
COPY ./workloads/autoware/control_validator_full_test $tmpdir/autoware/control_validator_full_test

# Download per-module input_bag (.db3) from Google Drive (too large for GitHub), keeping the original filename
RUN set -e; \
    fetch_bag() { mkdir -p "$1" && ( cd "$1" && gdown --fuzzy "$2" ); }; \
    fetch_bag "$tmpdir/autoware/control_validator_full_test/input_bag" 'https://drive.google.com/file/d/1QpWdfEJbCgougsI9Kgp7l1xFdV46cfUK/view?usp=sharing'; \
    fetch_bag "$tmpdir/autoware/crosswalk_velocity/input_bag" 'https://drive.google.com/file/d/10UxBivVJjZ34o3Ue_CVnVmLeHAXBNlZE/view?usp=sharing'; \
    fetch_bag "$tmpdir/autoware/euclidean_cluster/input_bag" 'https://drive.google.com/file/d/1Ep7YIUzi7Bn22yq13rRplRRAQp9JwOmM/view?usp=sharing'; \
    fetch_bag "$tmpdir/autoware/lane_change/input_bag" 'https://drive.google.com/file/d/1b0oXYEcxynHLX_2M7tYHLZc7NUGTFo-D/view?usp=sharing'; \
    fetch_bag "$tmpdir/autoware/mpc_controller_test/input_bag" 'https://drive.google.com/file/d/1G3AIBx-9Of_eLmkGRlea_sgoT4DSJ5Il/view?usp=sharing'; \
    fetch_bag "$tmpdir/autoware/path_optimizer/input_bag" 'https://drive.google.com/file/d/1HwWdLN77OfvsvXRbWy4f46uFp7zR34as/view?usp=sharing'; \
    fetch_bag "$tmpdir/autoware/pid_longitudinal_controller_test/input_bag" 'https://drive.google.com/file/d/1j5e4dHs2o3FZ093Vt_XxwSGadgBoTmyH/view?usp=sharing'

RUN cd $tmpdir/autoware/my_yabloc_pose_call_pkg && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select my_yabloc_pose_call_pkg && \
    cd $tmpdir/autoware/lane_detector && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select lane_detector && \
    cd $tmpdir/autoware/lane_change && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select lane_change && \
    cd $tmpdir/autoware/mpc_controller_test && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select mpc_controller_test && \
    cd $tmpdir/autoware/euclidean_cluster && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select euclidean_cluster && \
    cd $tmpdir/autoware/crosswalk_velocity && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select crosswalk_velocity && \
    cd $tmpdir/autoware/pid_longitudinal_controller_test && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select pid_longitudinal_controller_test && \
    cd $tmpdir/autoware/path_optimizer && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select path_optimizer && \
    cd $tmpdir/autoware/control_validator_full_test && source /opt/ros/$ROS_DISTRO/setup.bash && source $tmpdir/autoware/install/setup.bash && colcon build --packages-select control_validator_full_test

# Install driving_log_replayer via pipx
RUN apt-get remove -y python3-termcolor || true && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging && \
    python3 -m pip install --no-cache-dir git+https://github.com/tier4/driving_log_replayer.git

# Bash completion setup for driving_log_replayer
RUN _DLR_COMPLETE=bash_source dlr > $tmpdir/.dlr-complete.bash && \
    _DLR_COMPLETE=bash_source dlr > $tmpdir/.dlr-analyzer-complete.bash && \
    echo "source $tmpdir/.dlr-complete.bash" >> ~/.bashrc && \
    echo "source $tmpdir/.dlr-analyzer-complete.bash" >> ~/.bashrc

# Prepare maps in $tmpdir
RUN mkdir -p $tmpdir/autoware_map && \
    gdown -O $tmpdir/autoware_map/sample-map-planning.zip 'https://docs.google.com/uc?export=download&id=1499_nsbUbIeturZaDj7jhUownh5fvXHd' && \
    unzip -d $tmpdir/autoware_map $tmpdir/autoware_map/sample-map-planning.zip

# Additional map (nishishinjuku)
RUN wget -O $tmpdir/autoware_map/nishishinjuku_autoware_map.zip https://github.com/tier4/AWSIM/releases/download/v1.1.0/nishishinjuku_autoware_map.zip && \
    unzip -d $tmpdir/autoware_map $tmpdir/autoware_map/nishishinjuku_autoware_map.zip

# Driving log replayer data
RUN mkdir -p $tmpdir/driving_log_replayer_data

# annotationless_perception, localization, obstacle_segmentation, perception sample data
RUN gdown -O $tmpdir/driving_log_replayer_data/sample_dataset_v2.tar.zst 'https://docs.google.com/uc?export=download&id=1iCoykBBETI_rGfKEFYYb7LFydF-RJVkC' && \
    tar -I zstd -xvf $tmpdir/driving_log_replayer_data/sample_dataset_v2.tar.zst -C $tmpdir/driving_log_replayer_data/

# yabloc, eagleye, artag sample data
RUN gdown -O $tmpdir/driving_log_replayer_data/sample_bag.tar.zst 'https://docs.google.com/uc?export=download&id=17ppdMKi4IC8J_2-_9nyYv-LAfW0M1re5' && \
    tar -I zstd -xvf $tmpdir/driving_log_replayer_data/sample_bag.tar.zst -C $tmpdir/driving_log_replayer_data/

# Copy default config
RUN cp $tmpdir/autoware/src/simulator/driving_log_replayer/sample/.driving_log_replayer.config.toml ~/ && \
    cp $tmpdir/autoware/src/simulator/driving_log_replayer/sample/.driving_log_replayer.config.toml $tmpdir/

# Prepare yabloc scenario
RUN mkdir -p $tmpdir/driving_log_replayer_data/yabloc/sample && \
    cp -r $tmpdir/autoware/src/simulator/driving_log_replayer/sample/yabloc/scenario.yaml $tmpdir/driving_log_replayer_data/yabloc/sample && \
    cp -r $tmpdir/driving_log_replayer_data/sample_bag/yabloc/input_bag $tmpdir/driving_log_replayer_data/yabloc/sample

# Prepare simpoint result
RUN mkdir -p $tmpdir/simpoint_flow/trace_autoware/result_bag/traces/whole

COPY ./workloads/autoware/workload_user_entrypoint.sh /usr/local/bin/workload_user_entrypoint.sh

# Default command
CMD ["/bin/bash"]