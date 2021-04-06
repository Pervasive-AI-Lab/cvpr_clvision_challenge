#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: build_docker_image.sh SUBMISSION_DIR IMAGE_NAME"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "The provided submission directory doesn't exist!"
    exit 1
fi

SUBMISSION_DIR=$(cd "$1"; pwd)
IMAGE_NAME="$2"

if [ -f "$SUBMISSION_DIR/Dockerfile" ]; then
    echo "Building image $IMAGE_NAME from directory $SUBMISSION_DIR"
else
    echo "The provided submission directory is not valid: $SUBMISSION_DIR"
    exit 1
fi

cd "$SUBMISSION_DIR"

docker build -t "$IMAGE_NAME" .