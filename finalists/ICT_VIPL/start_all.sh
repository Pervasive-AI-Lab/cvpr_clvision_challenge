#!/usr/bin/env bash
set -e

echo "go!"
echo

echo "ni ..."
echo

cd /workspace/ni

python3.8 -u /workspace/ni/ni_replay_v3.py \
  --scenario=ni \
  --data_dir=/workspace/core50/data/ \
  --cls=WideResNet50 \
  --lr=0.01 \
  --batch_size=80 \
  --epochs=4 \
  --replay_examples=500 \
  --replay_strategy=random \
  --sub_dir=/workspace/submissions/ni \
  --lr_step=2 \
  --lr_gamma=0.5 \
  --aug=l3 \
  --random_seed=0

echo "multi-task-nc ..."
echo

cd /workspace/multi-task-nc

python3.8 -u /workspace/multi-task-nc/work_cmb.py \
  --scenario=multi-task-nc \
  --data_dir=/workspace/core50/data/ \
  --cls=ResNeXt50 \
  --epochs=1 \
  --replay_examples=3000 \
  --sub_dir=/workspace/submissions/multi-task-nc \
  --lr_1=0.01 \
  --lr_2=0.01 \
  --weight_decay=1e-5

echo "nic ..."
echo

cd /workspace/nic

python3.8 -u /workspace/nic/nic_v2.py --scenario=nic --sub_dir=/workspace/submissions/nic --submit_ver --data_dir=/workspace/core50/data/

cd /workspace

echo "completed!"
echo
