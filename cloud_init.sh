#!/bin/bash
set -euo pipefail

sudo apt update
sudo snap install aws-cli --classic

aws configure

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b
export PATH="$HOME/miniconda3/bin:$PATH"
conda init

pip install -r requirements.txt

export DATA_FOLDER="data"
export S3_BUCKET="ifera-marketdata"

# Ask for github token
read -p "Enter your GitHub token: " GITHUB_TOKEN
