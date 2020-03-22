#!/usr/bin/env bash
set -e

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
if [ "$#" -lt 1 ]; then
    echo "Usage: check_submission.sh PATH_TO_DATA_DIR [GPU_ID]"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "The provided data directory doesn't exist!"
    exit 1
fi

DATA_DIR=$(cd "$1"; pwd)
if [ ! -d "$1" ]; then
    echo "The provided data directory doesn't exist!"
    exit 1
fi

if [ -f "$DATA_DIR/labels.pkl" ]; then
    echo "Will use data dir $DATA_DIR"
else
    echo "The provided data directory is not valid!"
    exit 1
fi

# Allow the user to terminate the training process using Ctrl-C
if [ -t 0 ] ; then
    IS_INTERACTIVE_CMD="-it"
else
    IS_INTERACTIVE_CMD=""
fi

# Run in Docker
if [ -z "$2" ]
  then
    echo "Will expose all GPUs"
    docker run --gpus all \
        $IS_INTERACTIVE_CMD --rm \
        --user $(id -u):$(id -g) \
        -v "$PWD:/workspace" \
        -v "$DATA_DIR:/workspace/core50/data:ro" \
        cvpr_clvision_image bash create_submission.sh
else
    echo "Will use GPU(s) $2"
    docker run --gpus "\"device=$2\"" \
        $IS_INTERACTIVE_CMD --rm \
        --user $(id -u):$(id -g) \
        -v "$PWD:/workspace" \
        -v "$DATA_DIR:/workspace/core50/data:ro" \
        cvpr_clvision_image bash create_submission.sh
fi

set +e

# If you get an error regarding "--gpus all", try replacing the "Run in Docker" if/else block above with the following line:
#docker run --runtime=nvidia $IS_INTERACTIVE_CMD --rm --user $(id -u):$(id -g) -v "$PWD:/workspace" -v "$DATA_DIR:/workspace/core50/data:ro" cvpr_clvision_image bash create_submission.sh