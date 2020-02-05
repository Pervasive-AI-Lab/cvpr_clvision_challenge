# CVPR 2020 CLVision Challenge

This is the official starting repository for the CVPR 2020 CLVision 
challenge con *Continual Learning for Computer Vision*. We will provide:

- This github repository to help you with: 1) data loading & continual
learning protocols; 2) evaluation 3) generation of the submission file.
- Starting Dockerfile with everything pre-loaded for you.

You just have to write your own Continual Learning strategy and you
are ready to partecipate! 


### Challenge Description and Rules

You can find the challenge description and main rules in the official 
[workshop page](https://sites.google.com/view/clvision2020/challenge?authuser=0).


### Getting Started

load dataset:
```bash
./core50/scripts/bash/fetch_data_and_setup.sh
```
pytorch starter code:
```bash
python ./pytorch_main.py --scenario nicv2_79 --classifier ResNet18 --replay True
```
starter code:
```bash
python ./main.py
```
