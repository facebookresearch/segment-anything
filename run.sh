#!/bin/bash
set -e  # Exit on error

# Local image tag and container name
LOCAL_TAG="gs-segment-anything:local"
CONTAINER_NAME="gs-segment-anything"

# Remove any existing container with the same name
if docker ps -a --format '{{.Names}}' | grep -Eq "^${CONTAINER_NAME}\$"; then
  echo "🧹 Removing existing container: $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME"
fi

echo "🏃 Starting container in background..."
docker run \
  --name "$CONTAINER_NAME" \
  -e ORDER_ID="1640" \
  -e DATA_BUCKET="dev-golfstripes-ferrule-data" \
  -e AWS_PROFILE="golfstripes" \
  -v ~/.aws:/root/.aws \
  "$LOCAL_TAG"

#echo "🔗 Connecting to container shell..."
#docker exec -it "$CONTAINER_NAME" /bin/bash
#docker logs "$CONTAINER_NAME" -f
