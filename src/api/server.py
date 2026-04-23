import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from models.load_model import load_model
from utils.load_config import load_config

app = FastAPI(title="Image Classification Serving")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify ["http://localhost:5173"] for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Khởi tạo Global Variables (Chỉ load model 1 lần khi start server)
config = load_config('config.yaml')
IMG_SIZE = config['DATA']['IMG_SIZE'] or (224, 224)
CLASS_NAMES = config['CLASSNAME']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Giữ nguyên data_transforms từ predict.py của bạn
from torchvision import transforms
data_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomAdjustSharpness(5.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = load_model()
model.to(device)
model.eval()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)) -> dict:
    # 2. Nhận ảnh từ request
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 3. Tiền xử lý (Sử dụng đúng transform bạn đã định nghĩa)
    input_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    # 4. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        
    # 5. Trả về JSON (Online Prediction style)
    return {
        "filename": file.filename,
        "prediction": CLASS_NAMES[preds[0]],
        "confidence": float(prob[0][preds[0]])
    }
@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification API. Use /predict endpoint to classify images."}
# Để chạy: uvicorn src.serving.app:app --reload