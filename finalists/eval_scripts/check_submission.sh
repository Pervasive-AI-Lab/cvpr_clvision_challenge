#!/usr/bin/env bash
set -euo pipefail

# !!!! THIS IS A CUSTOM VERSION OF THE SCRIPT FOUND IN THE ORIGINAL CHALLENGE REPO !!!!


# If you are participating in the competition, please ignore this script as it will be used by organizers only.
# 
# This script will be used to check if the submitted source code works.
# It's very similar to "create_submission_in_docker.sh", but it also allows to specify where the core50/data can be found.
# The provided directory will be mounted inside the Docker container at "/workspace/core50/data" in read-only mode.
# >> !!! Compile the custom Dockerfile submitted by the challenge participant before running this script !!! <<
# >> (otherwise, you'll end up running the submission in a custom Docker image created by a different participant ;) ) <<
# 
# Requires Nvidia Docker: https://github.com/NVIDIA/nvidia-docker

# Check parameters
if [ "$#" -lt 3 ]; then
    echo "Usage: check_submission.sh PATH_TO_DATA_DIR IMAGE_NAME SUBMISSION_DIR [GPU_ID]"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "The provided data directory doesn't exist!"
    exit 1
fi

DATA_DIR=$(cd "$1"; pwd)

if [ -f "$DATA_DIR/labels.pkl" ]; then
    echo "Will use data dir $DATA_DIR"
else
    echo "The provided data directory is not valid!"
    exit 1
fi

# Get Docker image name
IMAGE_NAME="$2"
echo "Image name: $IMAGE_NAME"

if [ ! -d "$3" ]; then
    echo "The provided submission directory doesn't exist!"
    exit 1
fi

SUBMISSION_DIR=$(cd "$3"; pwd)

# Allow the user to terminate the training process using Ctrl-C
if [ -t 0 ] ; then
    IS_INTERACTIVE_CMD="-it"
else
    IS_INTERACTIVE_CMD=""
fi

# Get Gpu Id name
if [ "$#" -lt 4 ]; then
    GPU_OPTION="all"
    echo "Will expose all GPUs"
else
    if [ -z "$4" ]
        then
        GPU_OPTION="all"
        echo "Will expose all GPUs"
    else
        GPU_OPTION="\"device=$4\""
        echo "Will use GPU(s) $4"
    fi
fi

rm -rf "$SUBMISSION_DIR/cl_ext_mem"
rm -rf "$SUBMISSION_DIR/submissions"
mkdir "$SUBMISSION_DIR/cl_ext_mem"
mkdir "$SUBMISSION_DIR/submissions"

sudo ./drop_cache.sh

# Run in Docker
docker run --gpus "$GPU_OPTION" \
    $IS_INTERACTIVE_CMD --rm \
    --user $(id -u):$(id -g) \
    --shm-size="12g" \
    -v "$SUBMISSION_DIR:/workspace" \
    -v "$DATA_DIR:/workspace/core50/data:ro" \
    "$IMAGE_NAME" bash create_submission.sh

# If you get an error regarding "--gpus all", try replacing the "Run in Docker" block above with the following line:
#docker run --runtime=nvidia $IS_INTERACTIVE_CMD --rm --user $(id -u):$(id -g) --shm-size="12g" -v "$SUBMISSION_DIR:/workspace" -v "$DATA_DIR:/workspace/core50/data:ro" "$IMAGE_NAME" bash create_submission.sh
