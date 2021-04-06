#!/usr/bin/env bash
set -e

# Run experiments: the submission for each exp will be created in "/submissions"

python main_nc.py --scenario="multi-task-nc" --sub_dir="multi-task-nc" --epochs 1 --lr 0.0005 \
-cls ResNet50 --replay_examples 100 --batch_size 32 --scheduler_step 3 \
--distill_coef 0.1 --use_mle 1

python main_ni.py --scenario="ni" --sub_dir="ni" --epochs 5 --lr 0.0007 \
-cls ResNet50 --replay_examples 30 --batch_size 32 --scheduler_step 3 \
--distill_coef 0.1 --use_mle 1

# create zip file to submit to codalab: please note that the directories should
# have the names below depending on the challenge category you want to submit
# too (at least one)
cd submissions && zip -r ../submission.zip ./ni ./multi-task-nc

set +e

