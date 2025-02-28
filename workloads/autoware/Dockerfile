# syntax = edrevo/dockerfile-plus

FROM ghcr.io/autowarefoundation/autoware:universe-devel-0.39.1-amd64
SHELL ["/bin/bash", "-c"]

INCLUDE+ ./common/Dockerfile.common

WORKDIR $tmpdir

# TODO: Install required packages
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get install -y pipx && \
    add-apt-repository universe && \
    add-apt-repository multiverse && \
    apt-get update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-utils \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-vcstool \
      make \
      zstd \
    && rm -rf /var/lib/apt/lists/*
    
RUN rosdep init || true  && rosdep fix-permissions && rosdep update

WORKDIR /tmp/autoware_repos
RUN git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/autowarefoundation/autoware.git .
RUN git sparse-checkout set autoware.repos simulator.repos
RUN mv autoware.repos /autoware/
RUN mv simulator.repos /autoware/
RUN rm -rf /tmp/autoware_repos

COPY ./workloads/autoware/simulator_append.txt /autoware/

WORKDIR /autoware
RUN cat simulator_append.txt >> simulator.repos
RUN mkdir -p src

RUN vcs import src < autoware.repos
RUN vcs import src < simulator.repos

RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    apt-get update && \
    rosdep fix-permissions && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro $ROS_DISTRO && \
    source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build --base-paths src --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN pipx install git+https://github.com/tier4/driving_log_replayer.git

ENV PATH="/root/.local/bin:${PATH}"

#bash
RUN _DLR_COMPLETE=bash_source dlr > $HOME/.dlr-complete.bash
RUN _DLR_COMPLETE=bash_source dlr > $HOME/.dlr-analyzer-complete.bash

RUN echo "source $HOME/.dlr-complete.bash" >> ~/.bashrc
RUN echo "source $HOME/.dlr-analyzer-complete.bash" >> ~/.bashrc

RUN ln -s /autoware ~/autoware

#annotationless_perception, localization, obstacle_segmentation, perception
RUN mkdir -p ~/autoware_map
RUN gdown -O ~/autoware_map/sample-map-planning.zip 'https://docs.google.com/uc?export=download&id=1499_nsbUbIeturZaDj7jhUownh5fvXHd'
RUN unzip -d ~/autoware_map ~/autoware_map/sample-map-planning.zip

#yabloc, eagleye
RUN wget -O ~/autoware_map/nishishinjuku_autoware_map.zip https://github.com/tier4/AWSIM/releases/download/v1.1.0/nishishinjuku_autoware_map.zip
RUN unzip -d ~/autoware_map ~/autoware_map/nishishinjuku_autoware_map.zip

#annotationless_perception, localization, obstacle_segmentation, perception
RUN mkdir -p ~/driving_log_replayer_data
RUN gdown -O ~/driving_log_replayer_data/sample_dataset_v2.tar.zst 'https://docs.google.com/uc?export=download&id=1iCoykBBETI_rGfKEFYYb7LFydF-RJVkC'
RUN tar -I zstd -xvf ~/driving_log_replayer_data/sample_dataset_v2.tar.zst -C ~/driving_log_replayer_data/

#yabloc, eagleye, artag
RUN gdown -O ~/driving_log_replayer_data/sample_bag.tar.zst 'https://docs.google.com/uc?export=download&id=17ppdMKi4IC8J_2-_9nyYv-LAfW0M1re5'
RUN tar -I zstd -xvf ~/driving_log_replayer_data/sample_bag.tar.zst -C ~/driving_log_replayer_data/

RUN cp ~/autoware/src/simulator/driving_log_replayer/sample/.driving_log_replayer.config.toml ~/

RUN mkdir -p ~/driving_log_replayer_data/yabloc/sample
RUN cp -r ~/autoware/src/simulator/driving_log_replayer/sample/yabloc/scenario.yaml ~/driving_log_replayer_data/yabloc/sample

RUN cp -r ~/driving_log_replayer_data/sample_bag/yabloc/input_bag ~/driving_log_replayer_data/yabloc/sample

#Collecting simulation data
RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    source /autoware/install/setup.bash && \
    dlr simulation run -p yabloc -l play_rate:=0.5

# Start your application
CMD ["/bin/bash"]
