#!/bin/bash
# =============================================================================
# Matterport3D Simulator (MatterSim) Installation Script
# =============================================================================
# Installs the MatterSim Python binding for use with the mp3d conda environment.
#
# What this script does:
#   1. Clones + pins Matterport3DSimulator (+ nested pybind11); see lib/thirdparty.sh
#   2. Checks prerequisites (nvidia-smi, conda)
#   3. Creates the mp3d conda environment from scripts/install/envs/ac_mp3d.yaml
#   4. Installs system dependencies via apt-get
#   5. Builds MatterSim from source with EGL (GPU) or OSMesa (CPU) rendering
#   6. Sets up PYTHONPATH via conda activation hook
#   7. Verifies connectivity graphs are present
#   8. Validates the installation with a quick import test
#
# Usage:
#   bash scripts/install/install_ac_mp3d.sh [OPTIONS]
#
# Options:
#   --osmesa        Use OSMesa (CPU) rendering instead of EGL (GPU, default)
#   --skip-conda    Skip conda environment creation
#   --skip-deps     Skip apt-get system dependency installation
#   --status        Check installation status without changing anything
#   --help          Show this help message
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MP3D_DIR="$REPO_ROOT/third_party/Matterport3DSimulator"
ENV_NAME="ac-mp3d"
ENV_YAML="$SCRIPT_DIR/envs/ac_mp3d.yaml"

# Pinned-clone helper (Matterport3DSimulator was a git submodule until 2026-06-30;
# commit ID now lives in scripts/install/lib/thirdparty.sh).
source "$SCRIPT_DIR/lib/thirdparty.sh"

# =============================================================================
# Helpers
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }

check_command() {
    if ! command -v "$1" &>/dev/null; then
        return 1
    fi
    return 0
}

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"

    local ok=true

    # MatterSim source (cloned + pinned by init_submodules just above)
    if [[ -f "$MP3D_DIR/CMakeLists.txt" ]]; then
        print_success "Matterport3DSimulator source is present"
    else
        print_error "MatterSim source not cloned: $MP3D_DIR/CMakeLists.txt not found"
        echo ""
        echo "  Clone failed — check network / git access, then re-run this script."
        echo "  (Source is auto-cloned via scripts/install/lib/thirdparty.sh.)"
        echo ""
        ok=false
    fi

    # conda
    if check_command conda; then
        print_success "conda found: $(conda --version)"
    else
        print_error "conda is not available. Install Miniforge or Anaconda first."
        ok=false
    fi

    # nvidia-smi (warn only — OSMesa path is valid without GPU)
    if [[ "$USE_OSMESA" == true ]]; then
        print_info "OSMesa mode selected — GPU not required"
    else
        if check_command nvidia-smi; then
            print_success "nvidia-smi found: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        else
            print_warning "nvidia-smi not found. EGL rendering requires a GPU."
            echo "  If you are on a CPU-only machine, re-run with --osmesa."
        fi
    fi

    # cmake — the ac-mp3d env yaml provides cmake>=3.10 and build_mattersim uses
    # the env's cmake, so a MISSING system cmake is only a warning (the build no
    # longer depends on a system-wide cmake, which may be absent on fresh hosts).
    if check_command cmake; then
        print_success "cmake found: $(cmake --version | head -1)"
    else
        print_warning "system cmake not found — build will use the ac-mp3d env's cmake (from the env yaml)."
    fi

    # make
    if check_command make; then
        print_success "make found"
    else
        print_error "make is not installed. Install build-essential: sudo apt-get install build-essential"
        ok=false
    fi

    if [[ "$ok" == false ]]; then
        print_error "Prerequisite check failed. Fix the issues above and retry."
        exit 1
    fi

    echo ""
}

# =============================================================================
# Step 2: Create conda environment
# =============================================================================

create_conda_env() {
    print_header "Creating Conda Environment: $ENV_NAME"

    if ! [[ -f "$ENV_YAML" ]]; then
        print_error "Environment file not found: $ENV_YAML"
        echo "  Expected at scripts/install/envs/ac_mp3d.yaml"
        exit 1
    fi

    if conda env list | grep -qE "^${ENV_NAME}\s"; then
        print_warning "Conda environment '$ENV_NAME' already exists — skipping creation."
        print_info "To recreate: conda env remove -n $ENV_NAME && run this script again."
    else
        print_info "Creating conda environment from $ENV_YAML ..."
        conda env create -f "$ENV_YAML" --yes
        print_success "Conda environment '$ENV_NAME' created."
    fi
}

# =============================================================================
# Step 3: Install system dependencies
# =============================================================================

install_system_deps() {
    print_header "Installing System Dependencies"

    local pkgs=(
        libjsoncpp-dev
        libepoxy-dev
        libglm-dev
        libosmesa6
        libosmesa6-dev
        libglew-dev
        libopencv-dev
    )

    if ! check_command sudo; then
        print_warning "sudo not available — skipping apt-get."
        print_info "Install these packages manually if the build fails:"
        echo "  ${pkgs[*]}"
        return 0
    fi

    print_info "Running: sudo apt-get install -y ${pkgs[*]}"
    sudo apt-get install -y "${pkgs[@]}"
    print_success "System dependencies installed."
}

# =============================================================================
# Step 4: Fetch MatterSim source (clone + pin; init nested pybind11)
# =============================================================================

init_submodules() {
    print_header "Fetching MatterSim Source"

    # Clone + pin Matterport3DSimulator and init its nested pybind11 submodule.
    ensure_thirdparty Matterport3DSimulator
    print_success "Matterport3DSimulator ready (pinned; pybind11 initialized)."
}

# =============================================================================
# Step 5: Build MatterSim
# =============================================================================

build_mattersim() {
    print_header "Building MatterSim"

    local build_dir="$MP3D_DIR/build"
    local render_flag

    if [[ "$USE_OSMESA" == true ]]; then
        render_flag="-DOSMESA_RENDERING=ON"
        print_info "Render backend: OSMesa (CPU)"
    else
        render_flag="-DEGL_RENDERING=ON"
        print_info "Render backend: EGL (GPU)"
    fi

    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    # Resolve conda env paths for OpenCV/GLM headers and libraries
    local conda_prefix
    conda_prefix=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)" 2>/dev/null)

    print_info "Running cmake ..."
    # CMAKE_PREFIX_PATH: find OpenCV/GLM from conda env
    # -I conda include: GLM headers (glm/glm.hpp)
    # -DCV_LOAD_IMAGE_ANYDEPTH: OpenCV 4.x removed the old C constant
    # System jsoncpp (libjsoncpp-dev) is used for ABI compatibility
    # Use the ac-mp3d env's cmake (from the env yaml) so the build does not
    # depend on a system-wide cmake (which may be absent on fresh hosts).
    CMAKE_PREFIX_PATH="$conda_prefix" "$conda_prefix/bin/cmake" -S "$MP3D_DIR" -B "$build_dir" \
        "$render_flag" \
        -DPYTHON_EXECUTABLE="$conda_prefix/bin/python" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        "-DCMAKE_CXX_FLAGS=-std=c++11 -I${conda_prefix}/include -DCV_LOAD_IMAGE_ANYDEPTH=cv::IMREAD_ANYDEPTH"

    local nproc
    nproc=$(nproc 2>/dev/null || echo 4)
    print_info "Running make with -j${nproc} (MatterSimPython target only) ..."
    # LIBRARY_PATH: let linker find conda OpenCV libs
    # Build only MatterSimPython (tests and mattersim_main have extra compat issues)
    LIBRARY_PATH="$conda_prefix/lib" make -C "$build_dir" MatterSimPython -j"$nproc"

    # Verify .so was produced
    local so_file
    so_file=$(find "$build_dir" -name "MatterSim.cpython-*.so" 2>/dev/null | head -1)
    if [[ -n "$so_file" ]]; then
        print_success "Build succeeded: $so_file"
    else
        print_error "Build completed but MatterSim.cpython-*.so not found in $build_dir"
        exit 1
    fi
}

# =============================================================================
# Step 6: Setup PYTHONPATH via conda activation hook
# =============================================================================

setup_pythonpath() {
    print_header "Setting Up PYTHONPATH"

    # Resolve the conda prefix for $ENV_NAME
    local conda_prefix
    conda_prefix=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)" 2>/dev/null)
    if [[ -z "$conda_prefix" ]]; then
        print_error "Could not determine conda prefix for environment '$ENV_NAME'."
        exit 1
    fi

    local activate_dir="$conda_prefix/etc/conda/activate.d"
    local deactivate_dir="$conda_prefix/etc/conda/deactivate.d"
    local activate_script="$activate_dir/mp3d_env_vars.sh"
    local deactivate_script="$deactivate_dir/mp3d_env_vars.sh"

    mkdir -p "$activate_dir" "$deactivate_dir"

    cat > "$activate_script" <<ACTIVATE
#!/bin/bash
# Auto-generated by scripts/install/install_ac_mp3d.sh — do not edit manually.
export _OLD_PYTHONPATH_MP3D="\${PYTHONPATH:-}"
export PYTHONPATH="${MP3D_DIR}/build:\${PYTHONPATH:-}"
export MATTERPORT_DATA_DIR="\${MATTERPORT_DATA_DIR:-${REPO_ROOT}/data/mp3d/v1/scans}"
ACTIVATE

    cat > "$deactivate_script" <<DEACTIVATE
#!/bin/bash
# Auto-generated by scripts/install/install_ac_mp3d.sh — do not edit manually.
export PYTHONPATH="\${_OLD_PYTHONPATH_MP3D}"
unset _OLD_PYTHONPATH_MP3D
unset MATTERPORT_DATA_DIR
DEACTIVATE

    chmod +x "$activate_script" "$deactivate_script"
    print_success "Activation hook written to: $activate_script"

    # Also create a .pth file in site-packages so bare python (server mode) finds MatterSim
    local site_packages
    site_packages=$(conda run -n "$ENV_NAME" python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [[ -n "$site_packages" ]]; then
        echo "${MP3D_DIR}/build" > "$site_packages/mattersim.pth"
        print_success ".pth file written to: $site_packages/mattersim.pth"
    fi

    print_info "PYTHONPATH will include: ${MP3D_DIR}/build"
    print_info "MATTERPORT_DATA_DIR defaults to: ${REPO_ROOT}/data/mp3d/v1/scans"
}

# =============================================================================
# Step 7: Verify connectivity graphs
# =============================================================================

verify_connectivity() {
    print_header "Verifying Connectivity Graphs"

    local conn_dir="$MP3D_DIR/connectivity"

    if [[ -d "$conn_dir" ]]; then
        local graph_count
        graph_count=$(find "$conn_dir" -name "*_connectivity.json" 2>/dev/null | wc -l)
        if [[ "$graph_count" -gt 0 ]]; then
            print_success "Connectivity graphs found: $graph_count files in $conn_dir"
        else
            print_warning "connectivity/ directory exists but contains no *_connectivity.json files."
            print_info "These are bundled with the Matterport3DSimulator submodule."
            print_info "Try: git -C \"$MP3D_DIR\" checkout HEAD -- connectivity/"
        fi
    else
        print_warning "Connectivity directory not found: $conn_dir"
        print_info "These should be present in the submodule. Check:"
        print_info "  git -C \"$MP3D_DIR\" status"
    fi
}

# =============================================================================
# Step 8: Validate installation
# =============================================================================

validate_installation() {
    print_header "Validating Installation"

    print_info "Running import test inside '$ENV_NAME' environment ..."

    local result
    result=$(conda run -n "$ENV_NAME" python - <<'PYEOF' 2>&1
import sys, os
build_dir = os.environ.get("_MP3D_BUILD_DIR", "")
if build_dir:
    sys.path.insert(0, build_dir)

try:
    import MatterSim
    sim = MatterSim.Simulator()
    print("MatterSim imported successfully")
    print("MatterSim version:", getattr(MatterSim, "__version__", "unknown"))
except ImportError as e:
    print("IMPORT_FAILED:", e)
    sys.exit(1)
PYEOF
)

    if echo "$result" | grep -q "MatterSim imported successfully"; then
        print_success "MatterSim import test passed."
        echo "$result" | while IFS= read -r line; do
            print_info "  $line"
        done
    else
        print_error "MatterSim import test FAILED."
        echo "$result"
        echo ""
        print_info "Troubleshooting:"
        print_info "  1. Re-activate the environment: conda deactivate && conda activate $ENV_NAME"
        print_info "  2. Check PYTHONPATH: echo \$PYTHONPATH"
        print_info "  3. Confirm the .so exists: ls ${MP3D_DIR}/build/MatterSim*.so"
        exit 1
    fi
}

# =============================================================================
# Status check
# =============================================================================

show_status() {
    print_header "MatterSim Installation Status"

    # Submodule
    if [[ -f "$MP3D_DIR/CMakeLists.txt" ]]; then
        print_success "Submodule: present at $MP3D_DIR"
    else
        print_error "Submodule: NOT FOUND ($MP3D_DIR/CMakeLists.txt missing)"
    fi

    # Conda env
    if conda env list 2>/dev/null | grep -qE "^${ENV_NAME}\s"; then
        print_success "Conda env: '$ENV_NAME' exists"
    else
        print_warning "Conda env: '$ENV_NAME' NOT FOUND (run without --status to install)"
    fi

    # Build artifact
    local so_file
    so_file=$(find "$MP3D_DIR/build" -name "MatterSim.cpython-*.so" 2>/dev/null | head -1)
    if [[ -n "$so_file" ]]; then
        print_success "Build artifact: $so_file"
    else
        print_warning "Build artifact: NOT FOUND in $MP3D_DIR/build/"
    fi

    # Activation hook
    local conda_prefix
    conda_prefix=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)" 2>/dev/null || true)
    if [[ -n "$conda_prefix" ]]; then
        local hook="$conda_prefix/etc/conda/activate.d/mp3d_env_vars.sh"
        if [[ -f "$hook" ]]; then
            print_success "Activation hook: $hook"
        else
            print_warning "Activation hook: NOT FOUND at $hook"
        fi
    fi

    # Connectivity graphs
    local graph_count
    graph_count=$(find "$MP3D_DIR/connectivity" -name "*_connectivity.json" 2>/dev/null | wc -l)
    if [[ "$graph_count" -gt 0 ]]; then
        print_success "Connectivity graphs: $graph_count files"
    else
        print_warning "Connectivity graphs: NOT FOUND in $MP3D_DIR/connectivity/"
    fi

    # Import test
    if [[ -n "$so_file" ]] && conda env list 2>/dev/null | grep -qE "^${ENV_NAME}\s"; then
        print_info "Running quick import test ..."
        local build_dir
        build_dir="$(dirname "$so_file")"
        if conda run -n "$ENV_NAME" python -c "
import sys; sys.path.insert(0,'${build_dir}')
import MatterSim; MatterSim.Simulator()
print('ok')
" 2>/dev/null | grep -q "^ok$"; then
            print_success "Import test: PASSED"
        else
            print_warning "Import test: FAILED (run without --status for full output)"
        fi
    fi

    echo ""
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat <<'HELP'
Matterport3D Simulator (MatterSim) Installation Script

Usage: bash scripts/install/install_ac_mp3d.sh [OPTIONS]

Options:
  --osmesa        Use OSMesa (CPU) rendering instead of EGL (GPU, default)
  --skip-conda    Skip conda environment creation
  --skip-deps     Skip apt-get system dependency installation
  --status        Check installation status without changing anything
  --help          Show this help message

Quickstart (GPU machine):
  bash scripts/install/install_ac_mp3d.sh

CPU-only machine:
  bash scripts/install/install_ac_mp3d.sh --osmesa

Skip steps already done:
  bash scripts/install/install_ac_mp3d.sh --skip-conda --skip-deps

Check what is installed:
  bash scripts/install/install_ac_mp3d.sh --status

After install, activate the environment:
  conda activate ac-mp3d
  python -c "import MatterSim; print('ok')"

Data setup:
  Set MATTERPORT_DATA_DIR to the directory containing scene data:
    /path/to/data/mp3d/v1/scans/{scene_id}/{scene_id}.skybox_images.zip  (etc.)
  Default (set automatically on activation):
    $REPO_ROOT/data/mp3d/v1/scans
HELP
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Defaults
    USE_OSMESA=false
    SKIP_CONDA=false
    SKIP_DEPS=false
    STATUS_ONLY=false

    # Parse flags
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --osmesa)      USE_OSMESA=true;   shift ;;
            --skip-conda)  SKIP_CONDA=true;   shift ;;
            --skip-deps)   SKIP_DEPS=true;    shift ;;
            --status)      STATUS_ONLY=true;  shift ;;
            --help|-h)     show_help; exit 0  ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_help
                exit 1
                ;;
        esac
    done

    if [[ "$STATUS_ONLY" == true ]]; then
        show_status
        exit 0
    fi

    print_header "MatterSim Installation"
    echo "  Repo root : $REPO_ROOT"
    echo "  MatterSim : $MP3D_DIR"
    echo "  Conda env : $ENV_NAME"
    echo "  Render    : $(if [[ "$USE_OSMESA" == true ]]; then echo "OSMesa (CPU)"; else echo "EGL (GPU)"; fi)"
    echo ""

    init_submodules        # clone + pin Matterport3DSimulator (+ pybind11) before the prereq check
    check_prerequisites

    [[ "$SKIP_CONDA" == false ]] && create_conda_env
    [[ "$SKIP_DEPS"  == false ]] && install_system_deps

    build_mattersim
    setup_pythonpath
    verify_connectivity
    validate_installation

    print_header "Installation Complete"
    echo "  Next steps:"
    echo ""
    echo "    conda activate $ENV_NAME"
    echo "    python -c \"import MatterSim; print('MatterSim ready')\""
    echo ""
    echo "  Scene data:"
    echo "    Place Matterport3D scan folders under:"
    echo "    ${REPO_ROOT}/data/mp3d/v1/scans/{scene_id}/"
    echo ""
    echo "    Or override the path:"
    echo "    export MATTERPORT_DATA_DIR=/your/custom/path"
    echo ""
    echo "  Status check:"
    echo "    bash scripts/install/install_ac_mp3d.sh --status"
    echo ""
}

main "$@"
