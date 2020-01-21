#!/usr/bin/env bash

# Setup
DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir $DIR/../../data
mkdir $DIR/../data/logs
mkdir $DIR/../../data/snapshots


echo "Downloading Core50 (128x128 version)..."
wget --directory-prefix=$DIR'/../../data/' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip

echo "Unzipping Core50..."
unzip $DIR/../../data/core50_128x128.zip -d $DIR/../../data/

mv $DIR/../../data/core50_128x128/* $DIR/../../data/
