#!/usr/bin/env bash

# Run experiments: the submission for each exp will be created in "/submissions"
python naive_baseline.py --scenario="ni" --sub_dir="ni"
python naive_baseline.py --scenario="multi-task-nc" --sub_dir="multi-task-nc"
python naive_baseline.py --scenario="nic" --sub_dir="nic"

# create zip file to submit to codalab: please not that the directories should have the names below
# depending on the challenge category you want to submit too (at least one)
zip -r submission.zip ./submissions/ni ./submissions/multi-task-nc ./submissions/nic

