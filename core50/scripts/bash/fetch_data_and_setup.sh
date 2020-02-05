#!/usr/bin/env bash

# Setup
DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir -p $DIR/../../../cl_ext_mem
mkdir -p $DIR/../../../submissions

echo "Downloading Core50 dataset (train/validation set)..."
wget --directory-prefix=$DIR'/../../data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip

echo "Downloading challenge test set..."
wget --directory-prefix=$DIR'/../../data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_challenge_test.zip


echo "Unzipping data..."
unzip $DIR/../../data/core50_128x128.zip -d $DIR/../../data/
unzip $DIR/../../data/core50_challenge_test.zip -d $DIR/../../data/

mv $DIR/../../data/core50_128x128/* $DIR/../../data/
