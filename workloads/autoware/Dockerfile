# syntax = edrevo/dockerfile-plus

FROM ghcr.io/autowarefoundation/autoware:universe-devel-0.39.1-amd64
SHELL ["/bin/bash", "-c"]

INCLUDE+ ./common/Dockerfile.common

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
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update && \
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

RUN cd $tmpdir && git clone https://github.com/andikleen/pmu-tools.git

USER root
RUN apt-get update && apt-get -y install linux-tools-generic
RUN alias perf=$(find /usr/lib/linux-tools/*/perf | head -1)
RUN echo "alias perf=\"\$(find /usr/lib/linux-tools/*/perf | head -1)\"" >> /root/.bashrc

RUN rosdep init || true  && rosdep fix-permissions && rosdep update

# Clone the autoware repos into a temporary folder
WORKDIR /tmp/autoware_repos
RUN git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/autowarefoundation/autoware.git .
RUN git sparse-checkout set autoware.repos simulator.repos

# Move the .repos files into $tmpdir/autoware
RUN mkdir -p $tmpdir/autoware
RUN mv autoware.repos $tmpdir/autoware/
RUN mv simulator.repos $tmpdir/autoware/
RUN rm -rf /tmp/autoware_repos

# Copy simulator_append.txt to $tmpdir/autoware
COPY ./workloads/autoware/simulator_append.txt $tmpdir/autoware/
COPY ./workloads/autoware/test.sh $tmpdir/autoware/
RUN chmod +x $tmpdir/autoware/test.sh

WORKDIR $tmpdir/autoware
RUN cat simulator_append.txt >> simulator.repos
RUN mkdir -p src

# Import repos
RUN vcs import src < autoware.repos
RUN vcs import src < simulator.repos

# Build Autoware
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    apt-get update && \
    rosdep fix-permissions && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro $ROS_DISTRO && \
    source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --base-paths src --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Install driving_log_replayer via pipx
RUN apt-get remove -y python3-termcolor || true
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging
RUN python3 -m pip install --no-cache-dir git+https://github.com/tier4/driving_log_replayer.git

# Bash completion setup for driving_log_replayer
RUN _DLR_COMPLETE=bash_source dlr > $tmpdir/.dlr-complete.bash
RUN _DLR_COMPLETE=bash_source dlr > $tmpdir/.dlr-analyzer-complete.bash
RUN echo "source $tmpdir/.dlr-complete.bash" >> ~/.bashrc
RUN echo "source $tmpdir/.dlr-analyzer-complete.bash" >> ~/.bashrc

# Prepare maps in $tmpdir
RUN mkdir -p $tmpdir/autoware_map
RUN gdown -O $tmpdir/autoware_map/sample-map-planning.zip 'https://docs.google.com/uc?export=download&id=1499_nsbUbIeturZaDj7jhUownh5fvXHd'
RUN unzip -d $tmpdir/autoware_map $tmpdir/autoware_map/sample-map-planning.zip

# Additional map (nishishinjuku)
RUN wget -O $tmpdir/autoware_map/nishishinjuku_autoware_map.zip https://github.com/tier4/AWSIM/releases/download/v1.1.0/nishishinjuku_autoware_map.zip
RUN unzip -d $tmpdir/autoware_map $tmpdir/autoware_map/nishishinjuku_autoware_map.zip

# Driving log replayer data
RUN mkdir -p $tmpdir/driving_log_replayer_data

# annotationless_perception, localization, obstacle_segmentation, perception sample data
RUN gdown -O $tmpdir/driving_log_replayer_data/sample_dataset_v2.tar.zst 'https://docs.google.com/uc?export=download&id=1iCoykBBETI_rGfKEFYYb7LFydF-RJVkC'
RUN tar -I zstd -xvf $tmpdir/driving_log_replayer_data/sample_dataset_v2.tar.zst -C $tmpdir/driving_log_replayer_data/

# yabloc, eagleye, artag sample data
RUN gdown -O $tmpdir/driving_log_replayer_data/sample_bag.tar.zst 'https://docs.google.com/uc?export=download&id=17ppdMKi4IC8J_2-_9nyYv-LAfW0M1re5'
RUN tar -I zstd -xvf $tmpdir/driving_log_replayer_data/sample_bag.tar.zst -C $tmpdir/driving_log_replayer_data/

# Copy default config
RUN cp $tmpdir/autoware/src/simulator/driving_log_replayer/sample/.driving_log_replayer.config.toml ~/
RUN cp $tmpdir/autoware/src/simulator/driving_log_replayer/sample/.driving_log_replayer.config.toml $tmpdir/

# Prepare yabloc scenario
RUN mkdir -p $tmpdir/driving_log_replayer_data/yabloc/sample
RUN cp -r $tmpdir/autoware/src/simulator/driving_log_replayer/sample/yabloc/scenario.yaml $tmpdir/driving_log_replayer_data/yabloc/sample
RUN cp -r $tmpdir/driving_log_replayer_data/sample_bag/yabloc/input_bag $tmpdir/driving_log_replayer_data/yabloc/sample

# Collect simulation data
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    source $tmpdir/autoware/install/setup.bash && \
    env HOME="$tmpdir" dlr simulation run -p yabloc -l play_rate:=0.5

# Default command
CMD ["/bin/bash"]
