# Simple CV - Image Classification

A PyTorch-based image classification project for classifying images into three categories: **cat**, **duck**, and **panda**. This project provides a clean and modular structure for training and inference using deep learning models.

## Features

- 🔥 **PyTorch-based** - Built with PyTorch and TorchVision
- 🚀 **Easy to configure** - All settings managed through `config.yaml`
- 📊 **Multiple models** - Support for various CNN architectures (ResNet18 by default)
- 💾 **Weight management** - Automatic saving of best weights and epoch checkpoints
- 🎯 **Batch prediction** - Efficient batch inference on test images
- 🔧 **Modular design** - Clean separation of concerns with utility modules

## Project Structure

```
simple_cv/
├── data/
│   ├── train/          # Training data
│   │   ├── cat/
│   │   ├── duck/
│   │   └── panda/
│   └── val/            # Validation data
│       ├── cat/
│       ├── duck/
│       └── panda/
├── src/
│   ├── config.yaml     # Configuration file
│   ├── train.py        # Training script
│   ├── predict.py      # Prediction script
│   └── utils/          # Utility modules
│       ├── load_config.py
│       ├── load_data.py
│       ├── load_loss.py
│       ├── load_model.py
│       ├── load_optim.py
│       ├── predict_model.py
│       └── train_model.py
├── weights/            # Saved model weights
├── test_img/           # Test images for prediction
├── pyproject.toml      # Project dependencies
└── README.md
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. If you don't have `uv` installed:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
# Clone the repository
git clone http://gitlab.technica.vn/hieunp/image_classification.git
cd image_classification

# Install dependencies with uv
uv sync
```

Alternatively, you can use pip:

```bash
pip install torch torchvision pillow pyyaml matplotlib pandas scikit-learn
```

## Configuration

Edit `src/config.yaml` to customize your training:

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
  MODEL_NAME: resnet18    # Can use other models (resnet50, vgg16, etc.)
  NUMCLASS: 3
  EPOCHS: 30
  LEARNING_RATE: 1.0e-05
  LOSS_FUNCTION: CrossEntropyLoss
  OPTIM_FUNCTION: Adam
  CHECKPOINT: ''          # Path to pretrained weights (optional)

WEIGHT:
  SAVE_WEIGHT_PATH: ../weights
  SAVE_BEST: true
```

## Usage

### Training

Navigate to the `src` directory and run:

```bash
cd src
python train.py
```

The training script will:
- Load data from `data/train` and `data/val`
- Train the model for the specified number of epochs
- Save the best model weights to `weights/best.pt`
- Save epoch checkpoints to `weights/epoch_N.pt`
- Use GPU automatically if available

### Prediction

Run inference on test images:

```bash
cd src
python predict.py --test_path ../test_img --batch_predict 8
```

Arguments:
- `--test_path`: Path to the directory containing test images
- `--batch_predict`: Batch size for prediction

The predictions will be saved to `predict.csv` in the project root.

## Requirements

- Python >= 3.12
- PyTorch >= 2.10.0
- TorchVision >= 0.25.0
- Pillow >= 12.1.1
- PyYAML >= 6.0.3
- pandas >= 3.0.1
- matplotlib >= 3.10.8
- scikit-learn >= 1.8.0

## Model Training Details

- **Architecture**: ResNet18 (configurable)
- **Optimizer**: Adam with learning rate 1e-5
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: StepLR (decay by 0.1 every 7 epochs)
- **Image Size**: 224x224 pixels
- **Batch Size**: 16 (configurable)

## GPU Support

The project automatically detects and uses CUDA-enabled GPUs if available. Training will fall back to CPU if no GPU is detected.

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Project maintained by**: hieunp@technica.vn