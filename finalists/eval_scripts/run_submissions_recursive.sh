#!/usr/bin/env bash
set -euo pipefail

# This script has been used to evaluate all submissions.
# This scipt will take care of creating the docker image, running the submission, cleaning up the cache, etc.
# 
# Requires Nvidia Docker: https://github.com/NVIDIA/nvidia-docker

if [ "$#" -lt 2 ]; then
    echo "Usage: check_submission.sh SUBMISSIONS_DIR DATA_DIR [GPU_ID]"
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "The provided submissions directory doesn't exist!"
    exit 1
fi

SUBMISSIONS_DIR=$(cd "$1"; pwd)

echo "Submission directory: $SUBMISSIONS_DIR"

if [ ! -d "$2" ]; then
    echo "The provided submissions directory doesn't exist!"
    exit 1
fi

DATA_DIR=$(cd "$2"; pwd)

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

# Get Gpu Id name
if [ "$#" -lt 3 ]; then
    GPU_ID="all"
    echo "Will expose all GPUs"
else
    if [ -z "$3" ]
        then
        GPU_ID="all"
        echo "Will expose all GPUs"
    else
        GPU_ID="$3"
        echo "Will use GPU(s) $GPU_ID"
    fi
fi

for submission_path in "$SUBMISSIONS_DIR"/*; do
    [ -d "${submission_path}" ] || continue # if not a directory, skip
    submission_name="$(basename "${submission_path}")"
    echo "Checking submission $submission_name"

    submission_clean_name=${submission_name//[^[:alpha:]]/} # Alternative: echo "$submission_name" | tr -dc '[:alpha:]' | tr '[:upper:]' '[:lower:]'

    image_name="clvision_submission_${submission_clean_name,,}"

    if ./build_docker_image.sh "$submission_path" "$image_name"; then
        echo "Image creation completed"
        echo "Starting script for submission $submission_name"
        if ./check_submission.sh "$DATA_DIR" "$image_name" "$submission_path" "$GPU_ID"; then
            echo "Submission $submission_name completed successfully!"
        else
            echo "Submission $submission_name failed!"
        fi
    else
        echo "Image creation failed!"
    fi
    echo "Will sleep for 3 minutes to let the system refrigerate"
    sleep 3m
done