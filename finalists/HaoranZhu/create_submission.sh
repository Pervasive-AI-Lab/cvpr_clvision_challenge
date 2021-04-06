#!/usr/bin/env bash
set -e

# Run experiments: the submission for each exp will be created in "/submissions"
echo "run ni"
python  main_ewc.py --scenario="ni" --sub_dir="ni" --classifier="ResNet50" --epochs=20 --batch=16

echo "run mtc"
python  main_ewc.py --scenario="multi-task-nc" --sub_dir="multi-task-nc" --classifier="ResNet50" --epochs=20 --batch=16

echo "run nic"
python main_ewc.py --scenario="nic" --sub_dir="nic" --classifier="ResNet50" --epochs=20 --batch=16

# create zip file to submit to codalab: please note that the directories should
# have the names below depending on the challenge category you want to submit
# too (at least one)
cd submissions && sudo zip -r ../submission.zip ./ni ./multi-task-nc ./nic

set +e
