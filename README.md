# Simple CV - Image Classification

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*OmKhW6JSe8wpKt-n1XAJNw.gif" alt="Image classification demo" width="720" />
</p>

A compact PyTorch project for image classification with a FastAPI backend and a Vite + React frontend. It supports training, batch prediction, and serving a single-image prediction API.

## Highlights

- ResNet backbones with configurable head size from YAML.
- FastAPI inference endpoint with image validation.
- Batch CLI prediction that writes `predict.csv`.
- Docker Compose setup for backend + frontend.
- `uv`-based dependency management.

## Project Structure

```text
image_classification/
├── configs/
│   └── config.yaml
├── data/
│   ├── train/
│   └── val/
├── src/
│   ├── api/
│   │   └── server.py
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
├── weights/
│   └── best.pt
├── frontend/
│   ├── Dockerfile
│   └── src/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Requirements

- Python >= 3.12
- Node >= 20 (for local frontend dev)
- Docker (optional, for containers)

## Installation (Local)

The project uses `uv` for dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/HieuNguyen1905/image_classification.git
cd image_classification
uv sync
```

## Configuration

The configuration file is [configs/config.yaml](configs/config.yaml). It controls the model, data, and class names used for both training and inference.

```yaml
CLASSNAME:
- cane
- cavallo
- elefante
- farfalla
- gallina
- gatto
- mucca
- pecora
- ragno
- scoiattolo
DATA:
  DATA_DIR: ../data
  IMG_SIZE: [224, 224]
  BATCHSIZES: 16
  NUM_WORKERS: 4
MODEL:
  MODEL_NAME: resnet50
  NUMCLASS: 10
  CHECKPOINT: '../weights/best.pt'
  LOSS_FUNCTION: CrossEntropyLoss
  OPTIM_FUNCTION: Adam
  LEARNING_RATE: 1.0e-05
  EPOCHS: 30
WEIGHT:
  SAVE_WEIGHT_PATH: ../weights
  SAVE_BEST: true
```

Notes:

- `CLASSNAME` order must match the training order used to create `best.pt`.
- `CHECKPOINT` can be relative to the project root or an absolute path.
- `CONFIG_PATH` environment variable can be used to point to another config file.

## Run Locally

### Backend API

```bash
uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/
```

Predict one image:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/your/image.jpg"
```

Expected response format:

```json
{
  "filename": "image.jpg",
  "prediction": "gatto",
  "confidence": 0.98
}
```

### Frontend (Dev)

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Run with Docker Compose

From the project root:

```bash
docker compose up --build
```

Services:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`

Stop services:

```bash
docker compose down
```

## Dataset Layout

Your dataset should be organized like this:

```text
data/
├── train/
│   ├── class_a/
│   ├── class_b/
│   └── ...
└── val/
    ├── class_a/
    ├── class_b/
    └── ...
```

Each subfolder must contain images from exactly one class. Folder names should match the order in `CLASSNAME`.

## Training (Optional)

```bash
cd src
python pipelines/train.py
```

The training pipeline:

- Loads data from `data/train` and `data/val`.
- Fine-tunes the configured ResNet backbone.
- Saves checkpoints to `weights/` and `weights/best.pt` when `SAVE_BEST` is enabled.

## Batch Prediction (CLI)

```bash
cd src
python pipelines/predict.py --test_path ../test_img --batch_predict 8
```

Outputs are saved to `predict.csv` in the project root.

## Supported Backbones

- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

## Artifacts

- `weights/best.pt`: best checkpoint by validation F1.
- `weights/epoch_*.pt`: checkpoint per epoch.
- `predict.csv`: prediction output file.

Project maintained by hieu.nguyenphuc1905@gmail.com