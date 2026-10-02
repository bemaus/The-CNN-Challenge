# CNN Challenge: Small Image Dataset Classification

## ITCS 6169 – Assignment 1

This project explores how a pretrained convolutional neural network (EfficientNet-B0) performs under different image augmentations. It applies multiple transformation techniques (color jitter, random crops, rotation, random erasing and CutMix), which can be adjusted in a config file to find the best results.

## Setup

Install the requirements (a virtual environment is recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Pick a configuration from `CNN/configs/`:

- `baseline.yaml` – EfficientNet-B0 with no augmentation
- `light_aug.yaml` – light augmentation
- `best.yaml` – the best-performing configuration

You can also create your own by copying one of these and changing its parameters.

From the project root, run:

```bash
python CNN/evaluate.py --config CNN/configs/<CONFIG_NAME>.yaml
```

For example:

```bash
python CNN/evaluate.py --config CNN/configs/light_aug.yaml
```

Individual settings can be overridden without editing a file:

```bash
python CNN/evaluate.py --config CNN/configs/best.yaml --set training.num_epochs=5
```

## Running Evalaute.py

The file excacutes commands to provide the user image samples, graphs, and final predictions. 
These results will be sent through a pop-up window and the program will pause until the window is closed.
To finish the program, simply close all pop-up windows.

## Testing Models

Use test.ipynb to see every models accuracy.
