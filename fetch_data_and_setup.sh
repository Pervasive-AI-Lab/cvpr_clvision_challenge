#!/usr/bin/env bash

# Setup
DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir -p $DIR/cl_ext_mem
mkdir -p $DIR/submissions

echo "Downloading Core50 dataset (train/validation set)..."
wget --directory-prefix=$DIR'/core50/data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip
wget --directory-prefix=$DIR'/core50/data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz

echo "Downloading challenge test set..."
wget --directory-prefix=$DIR'/core50/data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_challenge_test.zip

echo "Unzipping data..."
unzip $DIR/core50/data/core50_128x128.zip -d $DIR/core50/data/
unzip $DIR/core50/data/core50_challenge_test.zip -d $DIR/core50/data/

mv $DIR/core50/data/core50_128x128/* $DIR/core50/data/
