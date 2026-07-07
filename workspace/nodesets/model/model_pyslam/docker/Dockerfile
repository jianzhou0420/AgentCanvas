# Headless pySLAM image for AgentCanvas server-mode use.
#
# Derived from upstream rosdocker `Dockerfile_pyslam` (Freda), stripped of the
# rocker / X11 / ROS / Qt / GUI dev scaffolding — we only ever run
# `Slam(headless=True)`. GPL-3.0 pySLAM is cloned at BUILD time into the image;
# it is never vendored into the AgentCanvas repo (the source stays behind the
# container boundary, same treatment as habitat_sim).
#
# Build path faithfulness: pySLAM's own `install_all.sh` does the whole job.
# Two things make it non-interactive inside docker:
#   1. `/.dockerenv` exists → install_all.sh skips its `keep_sudo_alive` prompt.
#   2. the build user has NOPASSWD sudo → install_all.sh's internal `sudo apt`
#      calls (pyenv build deps, etc.) run unattended.
# No conda in the base → install_all.sh takes the venv path and pyenv-builds the
# pinned Python 3.11.9 itself.
#
# CPU-only: no CUDA base, no nvidia runtime. The classic SLAM core (ORB2 + DBoW3)
# is pure CPU; learning backends install but run on CPU (fine for the Tier-1
# tracking-survival smoke). A CUDA variant is a later, GPU-gated follow-up.
#
# Build (rootless docker; see reference_rootless_docker memory):
#   export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock
#   docker build -t agentcanvas/pyslam:cpu -f Dockerfile .

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ── system + pySLAM build dependencies (the pySLAM-specific subset of upstream) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        tzdata ca-certificates git curl wget rsync sudo \
        cmake build-essential \
        python3-pip python3-dev python3-tk python3-sdl2 \
        libprotobuf-dev libeigen3-dev libopencv-dev libsuitesparse-dev \
        libglew-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# ── non-root build user with NOPASSWD sudo (upstream's pattern) ──
# install_all.sh runs its own `sudo apt` for pyenv build deps; this makes those
# unattended without running the whole build as root.
ARG USER=slam
ARG UID=1000
RUN useradd -m -s /bin/bash -u ${UID} ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

# Sentinel that tells pySLAM's installer it is inside a container (skip sudo-alive).
RUN touch /.dockerenv

# Our build fixes (see docker/patches + the model_pyslam design note).
COPY patches/ /tmp/pyslam-patches/

USER ${USER}
WORKDIR /home/${USER}
ENV USER=${USER}

# ── clone + build pySLAM (submodules carry the C++ core: g2o, pangolin, DBoW3, …) ──
RUN git clone --recursive --progress https://github.com/luigifreda/pyslam.git

# FIX 1 — cpp_core won't compile on AVX512 CPUs built with -march=native:
#   descriptor_helpers.h defines hsum256_ps under `#elif __AVX2__` (skipped when
#   __AVX512F__ is also set) but calls it from an independent `#if __AVX2__` block.
#   The patch decouples the ISA guards. Without it: "'hsum256_ps' not declared".
RUN cd pyslam && git apply /tmp/pyslam-patches/0001-avx512-avx2-hsum-guard.patch

# FIX 2 — GTSAM's nonlinear.cpp is a template bomb; install_gtsam.sh's `make -j$(nproc)`
#   OOM-kills cc1plus on many-core hosts (24 cores × several GB each > RAM). Cap the
#   apparent core count so every $(nproc) in pySLAM's build scripts stays sane.
RUN printf '#!/bin/sh\necho 6\n' | sudo tee /usr/local/bin/nproc >/dev/null \
    && sudo chmod +x /usr/local/bin/nproc

RUN cd pyslam && ./install_all.sh

# FIX 3 — expose the compiled modules on the venv path. pySLAM's editable install
#   covers the `pyslam` package but not the standalone .so dirs (cpp_core,
#   pyslam_utils, g2o), so `import cpp_core/pyslam_utils/g2o` would fail at runtime.
RUN printf '/home/slam/pyslam/cpp/lib\n/home/slam/pyslam/pyslam/slam/cpp/lib\n/home/slam/pyslam/thirdparty/g2opy/lib\n' \
    > /home/slam/.python/venvs/pyslam/lib/python3.11/site-packages/pyslam_cpp_libs.pth

# Activate the pyslam env for any interactive/entrypoint shell.
RUN echo 'source /home/'${USER}'/pyslam/pyenv-activate.sh' >> /home/${USER}/.bashrc

# XDG_RUNTIME_DIR shim (pyslam viz libs touch it even headless).
USER root
RUN printf '#!/bin/bash\nexport XDG_RUNTIME_DIR=/tmp/runtime-$(id -u)\nmkdir -p "$XDG_RUNTIME_DIR"\nchmod 700 "$XDG_RUNTIME_DIR"\nsource /home/'${USER}'/pyslam/pyenv-activate.sh 2>/dev/null || true\nexec "$@"\n' > /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh
USER ${USER}
WORKDIR /home/${USER}/pyslam

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
