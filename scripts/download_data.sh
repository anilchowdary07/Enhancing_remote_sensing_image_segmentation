#!/bin/bash
# Download datasets for EADF-Net experiments

set -e

DATA_DIR="./data"
mkdir -p $DATA_DIR

echo "============================================="
echo "EADF-Net Dataset Download Script"
echo "============================================="

# Download LoveDA dataset
echo ""
echo "1. Downloading LoveDA dataset..."
echo "   Source: https://zenodo.org/records/5706578"
echo ""

LOVEDA_DIR="${DATA_DIR}/LoveDA"
if [ -d "$LOVEDA_DIR" ]; then
    echo "   LoveDA already exists at ${LOVEDA_DIR}, skipping."
else
    mkdir -p $LOVEDA_DIR
    echo "   Downloading Train set..."
    curl -L --progress-bar -o "${DATA_DIR}/loveda_train.zip" \
        "https://zenodo.org/records/5706578/files/Train.zip?download=1"
    echo "   Downloading Val set..."
    curl -L --progress-bar -o "${DATA_DIR}/loveda_val.zip" \
        "https://zenodo.org/records/5706578/files/Val.zip?download=1"
    
    echo "   Extracting..."
    unzip -q "${DATA_DIR}/loveda_train.zip" -d "$LOVEDA_DIR"
    unzip -q "${DATA_DIR}/loveda_val.zip" -d "$LOVEDA_DIR"
    
    rm -f "${DATA_DIR}/loveda_train.zip" "${DATA_DIR}/loveda_val.zip"
    echo "   LoveDA downloaded to ${LOVEDA_DIR}"
fi

# Download FGGS dataset
echo ""
echo "2. Downloading FGGS dataset..."
echo "   Source: https://huggingface.co/datasets/XavierJiezou/ktda-datasets"
echo ""

FGGS_DIR="${DATA_DIR}/FGGS"
if [ -d "$FGGS_DIR" ]; then
    echo "   FGGS already exists at ${FGGS_DIR}, skipping."
else
    mkdir -p $FGGS_DIR
    echo "   Note: FGGS dataset requires manual download from HuggingFace."
    echo "   Please visit: https://huggingface.co/datasets/XavierJiezou/ktda-datasets"
    echo "   Download and extract to: ${FGGS_DIR}"
    echo ""
    echo "   Alternatively, use the HuggingFace CLI:"
    echo "   pip install huggingface_hub"
    echo "   huggingface-cli download XavierJiezou/ktda-datasets --local-dir ${FGGS_DIR}"
fi

echo ""
echo "============================================="
echo "Dataset setup complete!"
echo "============================================="
echo ""
echo "Directory structure:"
echo "  data/"
echo "  ├── LoveDA/"
echo "  │   ├── Train/"
echo "  │   │   ├── Urban/"
echo "  │   │   │   ├── images_png/"
echo "  │   │   │   └── masks_png/"
echo "  │   │   └── Rural/"
echo "  │   │       ├── images_png/"
echo "  │   │       └── masks_png/"
echo "  │   └── Val/"
echo "  │       ├── Urban/..."
echo "  │       └── Rural/..."
echo "  └── FGGS/"
echo "      ├── train/"
echo "      └── test/"
