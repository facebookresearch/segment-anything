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
LOCAL_TAG="gs-segment-anything:local"
LATEST_TAG="gs-segment-anything:latest"
ECR_REPO="842676010442.dkr.ecr.us-east-1.amazonaws.com/gs-segment-anything:latest"

if [[ "$MODE" == "local" ]]; then
  echo "🔧 Building Docker image for local use..."
  docker build -t $LOCAL_TAG .
  echo "✅ Local image built: $LOCAL_TAG"
else
  echo "🔧 Building Docker image for ECS/Fargate (linux/amd64)..."
  docker buildx build --platform linux/amd64 -t $LATEST_TAG .

  echo "🏷️ Tagging image for ECR..."
  docker tag $LATEST_TAG $ECR_REPO

  echo "🔐 Logging in to ECR..."
  aws ecr get-login-password --region us-east-1 --profile golfstripes \
    | docker login --username AWS --password-stdin 842676010442.dkr.ecr.us-east-1.amazonaws.com

  echo "🚀 Pushing image to ECR..."
  docker push $ECR_REPO

  echo "✅ Done! Image pushed to $ECR_REPO"
fi
