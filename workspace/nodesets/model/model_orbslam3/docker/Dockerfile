# agentcanvas/orbslam3 — ORB-SLAM3 + boost-python bindings for Container Launch.
#
# Recipe follows hello-binit/ORB_SLAM3 (patch-1) INSTALL.md — the Hello Robot
# fork pair verified on Ubuntu 22.04 / Python 3.10 — with two container
# adaptations: OpenCV comes from apt (4.5.4, satisfies ORB-SLAM3's >=4.4
# requirement; skips a ~30 min source build) and Pangolin is pinned to v0.9.1
# for reproducibility.
#
# The image carries ONLY the SLAM stack + the requirements-serve set.
# AgentCanvas framework code is NEVER baked in — it rides the read-only repo
# mount at run time (see design-docs/operations/container-launch.html).
#
# Build:  bash build.sh    (targets the rootless docker daemon)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates cmake build-essential pkg-config \
    libboost-dev libboost-serialization-dev libboost-python-dev \
    libssl-dev libeigen3-dev \
    libgl1-mesa-dev libglew-dev libepoxy-dev \
    libopencv-dev \
    python3 python3-dev python3-pip python3-numpy \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Pangolin (viewer dep of ORB-SLAM3; built headless-capable, pinned release)
RUN git clone --recursive --depth 1 --branch v0.9.1 \
        https://github.com/stevenlovegrove/Pangolin /opt/src/Pangolin && \
    cmake -B /opt/src/Pangolin/build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF /opt/src/Pangolin && \
    cmake --build /opt/src/Pangolin/build -j6 && \
    cmake --install /opt/src/Pangolin/build && ldconfig

# ORB-SLAM3 (Hello Robot fork — carries the Ubuntu 22.04 build fixes)
RUN git clone --depth 1 https://github.com/hello-binit/ORB_SLAM3 /opt/src/ORB_SLAM3 && \
    cd /opt/src/ORB_SLAM3 && ./build.sh && \
    cd build && make install && \
    cd ../Thirdparty/Sophus/build && make install && ldconfig

# Python bindings (boost-python; installs the `orbslam3` module)
RUN git clone --depth 1 https://github.com/hello-binit/ORB_SLAM3-PythonBindings \
        /opt/src/ORB_SLAM3-PythonBindings && \
    cmake -B /opt/src/ORB_SLAM3-PythonBindings/build \
        -DCMAKE_BUILD_TYPE=Release /opt/src/ORB_SLAM3-PythonBindings && \
    cmake --build /opt/src/ORB_SLAM3-PythonBindings/build -j6 && \
    cmake --install /opt/src/ORB_SLAM3-PythonBindings/build && ldconfig

# ORB vocabulary — part of the code, not a weight; baked in extracted form
RUN mkdir -p /opt/orbslam3 && \
    tar -xzf /opt/src/ORB_SLAM3/Vocabulary/ORBvoc.txt.tar.gz -C /opt/orbslam3

# Serve deps: lets the stock app.server.auto_host run from the repo mount
COPY requirements-serve.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-serve.txt

ENV LD_LIBRARY_PATH=/usr/local/lib
# Smoke at build time: the binding must import
RUN python3 -c "import orbslam3; print('orbslam3 binding OK')"
