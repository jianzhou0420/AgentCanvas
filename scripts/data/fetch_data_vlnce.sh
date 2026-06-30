#!/bin/bash
# =============================================================================
# VLN-CE Data Setup Script
# =============================================================================
# Downloads and sets up all required data for VLN-CE evaluation:
#   - R2R_VLNCE_v1-3_preprocessed dataset (required, 250MB)
#   - Matterport3D habitat scenes (required, ~15GB, needs ToS agreement)
#   - DDPPO depth encoder models (required for baselines, 672MB)
#   - RxR dataset (optional, for multilingual experiments)
#   - RxR BERT text features (optional, 142GB)
#   - Pretrained model checkpoints (optional)
#
# Usage:
#   ./fetch_data_vlnce.sh [OPTIONS]
#
# Options:
#   --all           Download everything (interactive for MP3D)
#   --r2r           Download R2R_VLNCE dataset (required for R2R experiments)
#   --mp3d          Download Matterport3D scenes (interactive, requires ToS)
#   --ddppo         Download DDPPO depth encoder models (672MB)
#   --rxr           Download RxR dataset
#   --rxr-bert      Download RxR BERT features (142GB, requires gsutil)
#   --checkpoints   Download pretrained checkpoints
#   --skip-mp3d     Skip MP3D download (use with --all)
#   --status        Show current data status
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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/habitat"    # was: $PROJECT_ROOT/data

# Google Drive file IDs (from README.md)
GDRIVE_R2R_FULL="1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm"
GDRIVE_R2R_PREPROCESSED="1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr"
GDRIVE_RXR_DATASET="145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t"
GDRIVE_CMA_EN="1fe0-w6ElGwX5VWtESKSM_20VY7sfn4fV"
GDRIVE_CMA_HI="1z84xMJ1LP2NO_jpJjFdymejXQqhU6zZH"
GDRIVE_CMA_TE="13mGjoKyJaWSJsnoQ-el4oIAlai0l7zfQ"

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
        print_error "$1 is not installed."
        return 1
    fi
    return 0
}

ensure_gdown() {
    if ! check_command gdown; then
        print_info "Installing gdown..."
        pip install gdown
    fi
}

download_gdrive() {
    local file_id="$1"
    local output_path="$2"
    local description="$3"

    if [[ -f "$output_path" ]]; then
        print_warning "$description already exists, skipping."
        return 0
    fi

    print_info "Downloading $description..."
    gdown "https://drive.google.com/uc?id=$file_id" -O "$output_path"

    if [[ -f "$output_path" ]]; then
        print_success "Downloaded $description"
    else
        print_error "Failed to download $description"
        return 1
    fi
}

# =============================================================================
# Download: R2R Dataset (required)
# =============================================================================

download_r2r() {
    print_header "R2R_VLNCE Dataset Download"

    ensure_gdown

    local r2r_dir="$DATA_DIR/datasets/R2R_VLNCE_v1-3_preprocessed"
    local r2r_zip="$DATA_DIR/datasets/R2R_VLNCE_v1-3_preprocessed.zip"

    if [[ -d "$r2r_dir" ]]; then
        print_warning "R2R dataset already exists at $r2r_dir"
        read -p "Re-download? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            return 0
        fi
    fi

    mkdir -p "$DATA_DIR/datasets"

    # Download preprocessed version (250MB, runs baselines out of the box)
    download_gdrive "$GDRIVE_R2R_PREPROCESSED" "$r2r_zip" "R2R_VLNCE_v1-3_preprocessed.zip (250MB)"

    if [[ -f "$r2r_zip" ]]; then
        print_info "Extracting R2R dataset..."
        unzip -o "$r2r_zip" -d "$DATA_DIR/datasets/"
        print_success "Extracted R2R dataset to $r2r_dir"
        rm -f "$r2r_zip"

        # Verify
        local split_count=$(find "$r2r_dir" -name "*.json.gz" 2>/dev/null | wc -l)
        print_info "Found $split_count split files"
    fi
}

# =============================================================================
# Download: MP3D Scenes (required)
# =============================================================================

download_mp3d() {
    print_header "Matterport3D Scenes Download"

    local mp3d_dir="$DATA_DIR/scene_datasets/mp3d"
    local scene_count=$(find "$mp3d_dir" -name "*.glb" 2>/dev/null | wc -l)

    if [[ $scene_count -gt 0 ]]; then
        print_warning "MP3D scenes already exist ($scene_count .glb files)"
        read -p "Re-download? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            return 0
        fi
    fi

    echo "Matterport3D scenes are REQUIRED for VLN-CE evaluation."
    echo ""
    echo "Before downloading, you must agree to the Matterport3D Terms of Use:"
    echo "  https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf"
    echo ""
    echo "This downloads only the Habitat task data (~15GB), not the full 1.3TB dataset."
    echo ""

    read -p "Have you read and agreed to the MP3D Terms of Use? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_warning "Skipping MP3D download. You must agree to ToS to proceed."
        return 0
    fi

    mkdir -p "$mp3d_dir"

    # Download mp3d_habitat.zip directly (15GB)
    # This avoids download.py which has interactive prompts and can
    # accidentally download the full 1.3TB dataset.
    local mp3d_url="http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip"
    local habitat_zip="$mp3d_dir/mp3d_habitat.zip"

    if [[ -f "$habitat_zip" ]]; then
        print_warning "mp3d_habitat.zip already downloaded, skipping."
    else
        print_info "Downloading mp3d_habitat.zip (~15GB)..."
        wget -q --show-progress "$mp3d_url" -O "$habitat_zip"

        if [[ ! -f "$habitat_zip" ]]; then
            print_error "Failed to download mp3d_habitat.zip"
            return 1
        fi
        print_success "Downloaded mp3d_habitat.zip"
    fi

    # Extract: zip contains mp3d/{scene}/{scene}.glb
    print_info "Extracting MP3D habitat data..."
    unzip -o "$habitat_zip" -d "$DATA_DIR/scene_datasets/"

    scene_count=$(find "$mp3d_dir" -name "*.glb" 2>/dev/null | wc -l)
    if [[ $scene_count -gt 0 ]]; then
        print_success "Extracted $scene_count scene files (.glb)"
        rm -f "$habitat_zip"
    else
        print_error "Extraction failed — no .glb files found"
        return 1
    fi
}

# =============================================================================
# Download: DDPPO Depth Encoder (required for baselines)
# =============================================================================

download_ddppo() {
    print_header "DDPPO Depth Encoder Models Download"

    local ddppo_dir="$DATA_DIR/ddppo-models"
    local ddppo_file="$ddppo_dir/gibson-2plus-resnet50.pth"

    if [[ -f "$ddppo_file" ]]; then
        print_warning "DDPPO models already exist at $ddppo_dir"
        read -p "Re-download? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            return 0
        fi
    fi

    mkdir -p "$ddppo_dir"

    print_info "Downloading ddppo-models.zip (672MB)..."
    local zip_file="$DATA_DIR/ddppo-models.zip"
    wget -q --show-progress \
        https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models.zip \
        -O "$zip_file"

    if [[ -f "$zip_file" ]]; then
        print_info "Extracting DDPPO models..."
        local temp_dir="$DATA_DIR/ddppo_temp"
        mkdir -p "$temp_dir"
        unzip -q "$zip_file" -d "$temp_dir"
        mv "$temp_dir"/data/ddppo-models/* "$ddppo_dir/"
        rm -rf "$temp_dir" "$zip_file"
        print_success "Extracted DDPPO models to $ddppo_dir"
    else
        print_error "Failed to download DDPPO models"
        return 1
    fi
}

# =============================================================================
# Download: RxR Dataset (optional)
# =============================================================================

download_rxr() {
    print_header "RxR Dataset Download"

    ensure_gdown

    local rxr_dir="$DATA_DIR/datasets/RxR_VLNCE_v0"
    local rxr_zip="$DATA_DIR/datasets/RxR_VLNCE_v0.zip"

    if [[ -d "$rxr_dir" ]]; then
        print_warning "RxR dataset already exists at $rxr_dir"
        read -p "Re-download? (y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            return 0
        fi
    fi

    mkdir -p "$DATA_DIR/datasets"

    download_gdrive "$GDRIVE_RXR_DATASET" "$rxr_zip" "RxR_VLNCE_v0.zip"

    if [[ -f "$rxr_zip" ]]; then
        print_info "Extracting RxR dataset..."
        unzip -o "$rxr_zip" -d "$DATA_DIR/datasets/"
        print_success "Extracted RxR dataset to $rxr_dir"
        rm -f "$rxr_zip"
    fi
}

# =============================================================================
# Download: RxR BERT Features (optional, 142GB)
# =============================================================================

download_rxr_bert() {
    print_header "RxR BERT Text Features Download"

    echo "This will download ~142GB of BERT text features using gsutil."
    echo "Features will be saved to: $DATA_DIR/datasets/RxR_VLNCE_v0/text_features/"
    echo ""

    check_command gsutil || {
        print_error "gsutil is required. Install Google Cloud SDK:"
        echo "  https://cloud.google.com/sdk/docs/install"
        return 1
    }

    read -p "Proceed with download? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        print_warning "Skipping BERT features download."
        return 0
    fi

    local bert_dir="$DATA_DIR/datasets/RxR_VLNCE_v0/text_features"
    mkdir -p "$bert_dir"

    print_info "Downloading BERT text features..."
    gsutil -m cp -R gs://rxr-data/text_features/* "$bert_dir/"

    print_success "Downloaded BERT text features"
}

# =============================================================================
# Download: Pretrained Checkpoints (optional)
# =============================================================================

download_checkpoints() {
    print_header "Pretrained Model Checkpoints Download"

    ensure_gdown

    local ckpt_dir="$DATA_DIR/checkpoints"
    mkdir -p "$ckpt_dir"

    echo "Available pretrained models:"
    echo "  1. CMA English (rxr_cma_en) - 196MB"
    echo "  2. CMA Hindi (rxr_cma_hi) - 196MB"
    echo "  3. CMA Telugu (rxr_cma_te) - 196MB"
    echo "  4. All of the above"
    echo ""

    read -p "Select option (1-4, or 0 to skip): " choice

    case $choice in
        1) download_gdrive "$GDRIVE_CMA_EN" "$ckpt_dir/rxr_cma_en.pth" "CMA English checkpoint" ;;
        2) download_gdrive "$GDRIVE_CMA_HI" "$ckpt_dir/rxr_cma_hi.pth" "CMA Hindi checkpoint" ;;
        3) download_gdrive "$GDRIVE_CMA_TE" "$ckpt_dir/rxr_cma_te.pth" "CMA Telugu checkpoint" ;;
        4)
            download_gdrive "$GDRIVE_CMA_EN" "$ckpt_dir/rxr_cma_en.pth" "CMA English checkpoint"
            download_gdrive "$GDRIVE_CMA_HI" "$ckpt_dir/rxr_cma_hi.pth" "CMA Hindi checkpoint"
            download_gdrive "$GDRIVE_CMA_TE" "$ckpt_dir/rxr_cma_te.pth" "CMA Telugu checkpoint"
            ;;
        0) print_warning "Skipping checkpoint download." ;;
        *) print_error "Invalid option"; return 1 ;;
    esac
}

# =============================================================================
# Status
# =============================================================================

show_status() {
    print_header "Current Data Status"

    # R2R Dataset
    local r2r_dir="$DATA_DIR/datasets/R2R_VLNCE_v1-3_preprocessed"
    if [[ -d "$r2r_dir" ]]; then
        local r2r_count=$(find "$r2r_dir" -name "*.json.gz" 2>/dev/null | wc -l)
        print_success "R2R Dataset: $r2r_count split files at $r2r_dir"
    else
        print_error "R2R Dataset: NOT FOUND (required — run with --r2r)"
    fi

    # RxR Dataset
    local rxr_dir="$DATA_DIR/datasets/RxR_VLNCE_v0"
    if [[ -d "$rxr_dir" ]]; then
        print_success "RxR Dataset: Found at $rxr_dir"
    else
        print_warning "RxR Dataset: NOT FOUND (optional)"
    fi

    # MP3D Scenes
    local mp3d_count=$(find "$DATA_DIR/scene_datasets/mp3d" -name "*.glb" 2>/dev/null | wc -l)
    if [[ $mp3d_count -gt 0 ]]; then
        print_success "MP3D Scenes: $mp3d_count .glb files"
    else
        print_error "MP3D Scenes: NOT FOUND (required — run with --mp3d)"
    fi

    # DDPPO Models
    if [[ -f "$DATA_DIR/ddppo-models/gibson-2plus-resnet50.pth" ]]; then
        print_success "DDPPO Models: Found"
    else
        print_warning "DDPPO Models: NOT FOUND (run with --ddppo)"
    fi

    # Checkpoints
    local ckpt_count=$(find "$DATA_DIR/checkpoints" -name "*.pth" 2>/dev/null | wc -l)
    if [[ $ckpt_count -gt 0 ]]; then
        print_success "Checkpoints: $ckpt_count files"
    else
        print_warning "Checkpoints: NOT FOUND (optional)"
    fi

    # RxR BERT Features
    local bert_dir="$DATA_DIR/datasets/RxR_VLNCE_v0/text_features"
    if [[ -d "$bert_dir" ]] && [[ "$(ls -A "$bert_dir" 2>/dev/null)" ]]; then
        print_success "RxR BERT Features: Found"
    else
        print_warning "RxR BERT Features: NOT FOUND (optional)"
    fi

    echo ""
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat <<'HELP'
VLN-CE Data Setup Script

Usage: ./fetch_data_vlnce.sh [OPTIONS]

Options:
  --all           Download everything (interactive for MP3D)
  --r2r           Download R2R_VLNCE_v1-3_preprocessed dataset (250MB)
  --mp3d          Download Matterport3D habitat scenes (~15GB)
  --ddppo         Download DDPPO depth encoder models (672MB)
  --rxr           Download RxR dataset
  --rxr-bert      Download RxR BERT features (142GB, requires gsutil)
  --checkpoints   Download pretrained checkpoints
  --skip-mp3d     Skip MP3D download (use with --all)
  --status        Show current data status
  --help          Show this help message

Quickstart (minimum for R2R eval):
  ./fetch_data_vlnce.sh --r2r --mp3d --ddppo

Examples:
  ./fetch_data_vlnce.sh --status               # Check what's installed
  ./fetch_data_vlnce.sh --r2r                   # Download R2R dataset only
  ./fetch_data_vlnce.sh --all --skip-mp3d       # Download everything except MP3D
  ./fetch_data_vlnce.sh --all                   # Download everything
HELP
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "VLN-CE Data Setup"

    echo "Project root: $PROJECT_ROOT"
    echo "Data directory: $DATA_DIR"
    echo ""

    local do_r2r=false
    local do_mp3d=false
    local do_ddppo=false
    local do_rxr=false
    local do_rxr_bert=false
    local do_checkpoints=false
    local skip_mp3d=false
    local show_status_only=false

    if [[ $# -eq 0 ]]; then
        show_help
        return 0
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                do_r2r=true; do_mp3d=true; do_ddppo=true
                do_rxr=true; do_checkpoints=true
                shift ;;
            --r2r)         do_r2r=true; shift ;;
            --mp3d)        do_mp3d=true; shift ;;
            --ddppo)       do_ddppo=true; shift ;;
            --rxr)         do_rxr=true; shift ;;
            --rxr-bert)    do_rxr_bert=true; shift ;;
            --checkpoints) do_checkpoints=true; shift ;;
            --skip-mp3d)   skip_mp3d=true; shift ;;
            --status)      show_status_only=true; shift ;;
            --help|-h)     show_help; return 0 ;;
            *) print_error "Unknown option: $1"; show_help; return 1 ;;
        esac
    done

    if [[ "$show_status_only" == true ]]; then
        show_status
        return 0
    fi

    # Execute downloads in dependency order
    [[ "$do_r2r" == true ]]                              && download_r2r
    [[ "$do_mp3d" == true && "$skip_mp3d" == false ]]    && download_mp3d
    [[ "$do_ddppo" == true ]]                            && download_ddppo
    [[ "$do_rxr" == true ]]                              && download_rxr
    [[ "$do_rxr_bert" == true ]]                         && download_rxr_bert
    [[ "$do_checkpoints" == true ]]                      && download_checkpoints

    show_status

    print_header "Setup Complete"
    echo "Next steps:"
    echo "  conda activate ac-vlnce"
    echo "  python -m env_runner.eval_mock --policy random --split val_seen --episodes 5"
    echo ""
}

main "$@"
