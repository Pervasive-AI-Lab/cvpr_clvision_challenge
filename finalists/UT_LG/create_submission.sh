#!/usr/bin/env bash
set -e

# Run experiments: the submission for each exp will be created in "/submissions"
python final_submission.py --parameters config/final/nc.yml #--verbose True
python final_submission.py --parameters config/final/ni.yml #--verbose True
python final_submission.py --parameters config/final/nic.yml #--verbose True


# create zip file to submit to codalab: please note that the directories should
# have the names below depending on the challenge category you want to submit
# too (at least one)
cd submissions && zip -r ../submission.zip ./ni ./multi-task-nc ./nic

set +e