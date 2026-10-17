#!/bin/bash
# Creates the virtual environment, installs training dependencies and downloads HIT-UAV.
set -e

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate

pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ -z "$KAGGLE_API_TOKEN" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    echo ""
    echo "ERROR: no Kaggle credentials found."
    echo "Get your API token: https://www.kaggle.com/settings → API → Create New Token"
    echo "Then either export KAGGLE_API_TOKEN=<token> or copy kaggle.json to ~/.kaggle/"
    exit 1
fi
[ -f "$HOME/.kaggle/kaggle.json" ] && chmod 600 "$HOME/.kaggle/kaggle.json"

if [ ! -d "dataset/hit-uav" ]; then
    echo "Downloading HIT-UAV dataset..."
    kaggle datasets download -d pandrii000/hituav-a-highaltitude-infrared-thermal-dataset
    mkdir -p dataset
    unzip -q hituav-a-highaltitude-infrared-thermal-dataset.zip -d dataset/
    rm hituav-a-highaltitude-infrared-thermal-dataset.zip
    echo "Dataset ready in dataset/hit-uav"
else
    echo "Dataset already exists, skipping."
fi

echo ""
echo "Setup complete. Next: make preprocess && make train"
