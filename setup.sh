#! /usr/bin/bash

sudo apt update

sudo apt install -y ffmpeg python3-pip

# CuDNN (if GPU available)
sudo apt install -y libcudnn8 libcudnn8-dev

pip install -r requirements.txt

mkdir data
python3 dataset_setup.py ./data