#!/bin/bash
set -e  # Exit on error

MODE=$1

MODEL_FILE="sam_vit_h_4b8939.pth"
MODEL_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Download the model weights if not already present
if [ ! -f "$MODEL_FILE" ]; then
  echo "📦 Model file not found, downloading..."
  curl -L -o "$MODEL_FILE" "$MODEL_URL"
else
  echo "✅ Model file already exists, skipping download."
fi

# Image tags
LOCAL_TAG="gs-segment-anything:mask"
echo "🔧 Building Docker image for mask use..."
docker build -t $LOCAL_TAG .
echo "✅ Local image built: $LOCAL_TAG"
