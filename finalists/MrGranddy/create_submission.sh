#!/usr/bin/env bash
set -e

# Run experiments: the submission for each exp will be created in "/submissions"
python naive_baseline.py --classifier="EfficientNet-B7" --scenario="ni" --sub_dir="ni" --batch_size="32" --epochs="6" --replay_examples="2000" --lr="0.0256"
python naive_baseline.py --classifier="EfficientNet-B7" --scenario="multi-task-nc" --sub_dir="multi-task-nc" --batch_size="32" --epochs="1" --replay_examples="2000" --lr="0.0256"

# create zip file to submit to codalab: please note that the directories should
# have the names below depending on the challenge category you want to submit
# too (at least one)
cd submissions && zip -r ../submission.zip ./ni ./multi-task-nc

set +e