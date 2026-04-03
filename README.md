# Simple CV - Image Classification

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*OmKhW6JSe8wpKt-n1XAJNw.gif" alt="Image classification demo" width="720" />
</p>

A compact PyTorch project for classifying images into three classes: **cat**, **duck**, and **panda**. The codebase is organized for training, prediction, and configuration management, making it suitable for quick experiments or as a base for a more complete image classification pipeline.

## Highlights

- Training and inference with PyTorch and TorchVision.
- Support for multiple ResNet backbones, with `resnet50` as the default.
- Automatic checkpoint saving by epoch, plus the best model when `SAVE_BEST` is enabled.
- Batch prediction for efficient inference on multiple images.
- Centralized configuration via YAML.

## Project Structure

```text
image_classification/
├── configs/
│   └── config.yaml
├── data/
│   ├── train/
│   │   ├── cat/
│   │   ├── duck/
│   │   └── panda/
│   └── val/
│       ├── cat/
│       ├── duck/
│       └── panda/
├── src/
│   ├── datasets/
│   │   └── load_data.py
│   ├── loss/
│   │   └── load_loss.py
│   ├── models/
│   │   └── load_model.py
│   ├── pipelines/
│   │   ├── train.py
│   │   └── predict.py
│   ├── training/
│   │   ├── load_optim.py
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── utils/
│       └── load_config.py
├── pyproject.toml
└── README.md
```

## Installation

The project uses `uv` for dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/HieuNguyen1905/image_classification.git
cd image_classification
uv sync
```

If you prefer `pip`, install the main dependencies manually:

```bash
pip install torch torchvision pillow pyyaml pandas matplotlib scikit-learn
```

## Dataset Setup

Your dataset should be organized like this:

```text
data/
├── train/
│   ├── cat/
│   ├── duck/
│   └── panda/
└── val/
    ├── cat/
    ├── duck/
    └── panda/
```

Each subfolder must contain images from exactly one class.

## Configuration

The configuration file is located at [configs/config.yaml](configs/config.yaml).

```yaml
CLASSNAME:
  - cat
  - duck
  - panda

DATA:
  DATA_DIR: ../data
  IMG_SIZE: [224, 224]
  BATCHSIZES: 16
  NUM_WORKERS: 4

MODEL:
  MODEL_NAME: resnet50
  NUMCLASS: 3
  EPOCHS: 30
  LEARNING_RATE: 1.0e-05
  LOSS_FUNCTION: CrossEntropyLoss
  OPTIM_FUNCTION: Adam
  CHECKPOINT: ''

WEIGHT:
  SAVE_WEIGHT_PATH: ../weights
  SAVE_BEST: true
```

Important: the current scripts read `config.yaml` from the working directory. The safest way to run the project is to copy the config file into `src/` before training or predicting:

```bash
cp configs/config.yaml src/config.yaml
```

## Training

Run the training script from the `src` directory:

```bash
cd src
python pipelines/train.py
```

The training pipeline will:

- Load data from `data/train` and `data/val`.
- Fine-tune the selected model from `config.yaml`.
- Save checkpoints for each epoch in the `weights/` directory.
- Save the best model to `weights/best.pt` when `SAVE_BEST` is enabled.
- Automatically use GPU if one is available.

## Prediction

Run inference on a single image or a directory of images:

```bash
cd src
python pipelines/predict.py --test_path ../test_img --batch_predict 8
```

Arguments:

- `--test_path`: path to an image file or a directory of images.
- `--batch_predict`: batch size used during inference.

The predictions will be saved to `predict.csv` in the project root.

## Model and Training Setup

- Supported backbones: `resnet18`, `resnet34`, `resnet50`, `resnet101`, and `resnet152`.
- Default loss: `CrossEntropyLoss`.
- Default optimizer: `Adam`.
- Scheduler: `StepLR`, reducing the learning rate every 7 epochs by a factor of `0.1`.
- Default image size: `224 x 224`.

## Generated Artifacts

- `weights/best.pt`: best checkpoint by validation F1.
- `weights/epoch_*.pt`: checkpoint for each epoch.
- `predict.csv`: prediction output file.

## Requirements

- Python >= 3.12
- PyTorch >= 2.10.0
- TorchVision >= 0.25.0
- Pillow >= 12.1.1
- PyYAML >= 6.0.3
- pandas >= 3.0.1
- matplotlib >= 3.10.8
- scikit-learn >= 1.8.0


Project maintained by hieu.nguyenphuc1905@gmail.com