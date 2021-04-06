#!/usr/bin/env bash
set -e

# Run experiments: the submission for each exp will be created in "/submissions"
python test.py --scenario="ni" --sub_dir="ni" --batch_size=16 --classifier='ResNetst' --ER_type='BER' --drl_lmb=0.001 --fix='1' --replay_examples=1500  
python test.py --scenario="multi-task-nc" --sub_dir="multi-task-nc" --batch_size=16 --classifier='ResNetst' --ER_type='ER' --drl_lmb=0.0002 --fix='1' --replay_examples=500
python test.py --scenario="nic" --sub_dir="nic" --batch_size=32 --classifier='ResNetst' --drl_lmb=0.00001 --fix='12' --replay_examples=50

# create zip file to submit to codalab: please note that the directories should
# have the names below depending on the challenge category you want to submit
# too (at least one)
cd submissions && zip -r ../submission.zip ./ni ./multi-task-nc ./nic

set +e