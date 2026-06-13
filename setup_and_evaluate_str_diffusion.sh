#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set environment variables to avoid home directory permission issues on Linux server
export TORCH_EXTENSIONS_DIR="/tmp/cks/.cache/torch_extensions"
export MPLCONFIGDIR="/tmp/cks/.cache/matplotlib"
export HF_HOME="/tmp/cks/.cache/huggingface"

# Automatically run inside a GNU screen session named 'strdiffusion' if screen is available
if [ -z "$STY" ]; then
    if command -v screen >/dev/null 2>&1; then
        echo "[*] Re-running script inside a GNU screen session named 'strdiffusion'..."
        # Wrap execution in bash -c so that screen stays open even if the script fails or finishes
        exec screen -S strdiffusion bash -c "bash \"$0\" \"$@\"; echo '------------------------------------------------'; echo 'Script finished or encountered an error. Screen session kept open.'; exec bash"
    else
        echo "[!] screen command not found. Proceeding in current session..."
    fi
fi

echo "====================================================================="
echo "      StrDiffusion Environment Setup & Evaluation Script"
echo "====================================================================="

# Ensure we are in the StrDiffusion repository folder
if [ ! -f "test/texture/config/inpainting/test.py" ]; then
    if [ -d "StrDiffusion" ]; then
        echo "[*] Moving into StrDiffusion folder..."
        cd StrDiffusion
    else
        echo "Error: Cannot find StrDiffusion directory. Please run this script inside the repository root."
        exit 1
    fi
fi

# Create python virtual environment
ENV_NAME="strdiffusion-env"
if [ ! -d "$ENV_NAME" ]; then
    echo "[*] Creating virtual environment: $ENV_NAME..."
    python3 -m venv "$ENV_NAME"
else
    echo "[*] Virtual environment $ENV_NAME already exists."
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source "$ENV_NAME"/bin/activate

# Install dependencies
echo "[*] Upgrading pip..."
pip install --upgrade pip

echo "[*] Installing PyTorch with CUDA 12.1 compatibility..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[*] Installing StrDiffusion pipeline requirements..."
pip install numpy opencv-python Pillow PyYAML scipy timm tqdm scikit-image gdown einops lmdb lpips tensorboardX torchsummaryX matplotlib ema-pytorch ninja

# Create checkpoint directory
CHECKPOINT_DIR="test/texture/config/inpainting/checkpoint"
mkdir -p "$CHECKPOINT_DIR"

# Download model checkpoints if missing
echo "[*] Downloading pretrained weights using gdown..."

if [ ! -f "$CHECKPOINT_DIR/t.pth" ]; then
    echo "[*] Downloading Texture Denoising Model (t.pth)..."
    gdown --no-cookies 1jKJYzC7uLsdYnpXi8nXr_EoBppeNZPKL -O "$CHECKPOINT_DIR/t.pth"
else
    echo "[*] Texture Denoising Model (t.pth) already exists."
fi

if [ ! -f "$CHECKPOINT_DIR/s.pth" ]; then
    echo "[*] Downloading Structure Denoising Model (s.pth)..."
    gdown --no-cookies 1efuBUlZFTE0V75Xx-zin1ZRZebP_1x0B -O "$CHECKPOINT_DIR/s.pth"
else
    echo "[*] Structure Denoising Model (s.pth) already exists."
fi

if [ ! -f "$CHECKPOINT_DIR/dis.pth" ]; then
    echo "[*] Downloading Discriminator Model (dis.pth)..."
    gdown --no-cookies 1wz0I_F66KFHOKSXTyKTlzFH6vV-t_Hrd -O "$CHECKPOINT_DIR/dis.pth"
else
    echo "[*] Discriminator Model (dis.pth) already exists."
fi

# Define default paths (user server paths)
DEFAULT_IMG_DIR="/tmp/cks/SEM-Net/datasets/places365/test_256"
DEFAULT_MASK_DIR="/tmp/cks/SEM-Net/datasets/testing_mask_dataset"
DEFAULT_OUT_DIR="./results"

# Ask user if they want to override paths
echo ""
echo "---------------------------------------------------------------------"
echo "Please confirm or specify dataset paths for evaluation:"
read -p "Enter Places365 images directory [$DEFAULT_IMG_DIR]: " IMG_DIR
IMG_DIR=${IMG_DIR:-$DEFAULT_IMG_DIR}

read -p "Enter Masks directory [$DEFAULT_MASK_DIR]: " MASK_DIR
MASK_DIR=${MASK_DIR:-$DEFAULT_MASK_DIR}

read -p "Enter Output directory [$DEFAULT_OUT_DIR]: " OUT_DIR
OUT_DIR=${OUT_DIR:-$DEFAULT_OUT_DIR}
echo "---------------------------------------------------------------------"

echo "[*] Running StrDiffusion model validation for the 5 target images..."
python test/texture/config/inpainting/validate_str_diffusion.py \
    --image_dir "$IMG_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUT_DIR" \
    --texture_ckpt "$CHECKPOINT_DIR/t.pth" \
    --structure_ckpt "$CHECKPOINT_DIR/s.pth" \
    --discriminator_ckpt "$CHECKPOINT_DIR/dis.pth"

echo "====================================================================="
echo "                    Evaluation Completed!"
echo "====================================================================="
