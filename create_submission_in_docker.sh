#!/usr/bin/env bash
set -e

# This script should be used to check if the submission source code can be run in your customized "cvpr_clvision_image" Docker image.
# In order for this to work, build your custom "cvpr_clvision_image" image using the "build_docker_image.sh" script.
# Please check the README for a more detailed explanation.
# 
# Requires Nvidia Docker: https://github.com/NVIDIA/nvidia-docker

# Allow the user to terminate the training process using Ctrl-C
if [ -t 0 ] ; then
    IS_INTERACTIVE_CMD="-it"
else
    IS_INTERACTIVE_CMD=""
fi

# Run in Docker
if [ -z "$1" ]
  then
    echo "Will expose all GPUs"
    docker run --gpus all \
      $IS_INTERACTIVE_CMD --rm \
      --user $(id -u):$(id -g) \
      -v "$PWD:/workspace" \
      cvpr_clvision_image bash create_submission.sh
else
    echo "Will use GPU $1"
    docker run --gpus "\"device=$1\"" \
      $IS_INTERACTIVE_CMD --rm \
      --user $(id -u):$(id -g) \
      -v "$PWD:/workspace" \
      cvpr_clvision_image bash create_submission.sh
fi

set +e

# If you get an error regarding "--gpus all", try replacing the "Run in Docker" if/else block above with the following line:
#docker run --runtime=nvidia $IS_INTERACTIVE_CMD --rm --user $(id -u):$(id -g) -v "$PWD:/workspace" cvpr_clvision_image bash create_submission.sh