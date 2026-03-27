import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from utils.load_config import load_config

cudnn.benchmark = True

import warnings
warnings.filterwarnings('ignore')   

def load_data():
    # load config
    config = load_config('config.yaml')
    IMG_SIZE = config['DATA']['IMG_SIZE'] if config['DATA']['IMG_SIZE'] else (224, 224)
    DATA_DIR = config['DATA']['DATA_DIR'] if config['DATA']['DATA_DIR'] else '../data'
    BATCHSIZE = config['DATA']['BATCHSIZES'] if config['DATA']['BATCHSIZES'] else 16
    NUM_WORKERS = config['DATA']['NUM_WORKERS'] if config['DATA']['NUM_WORKERS'] else 4

    # transforms for dataset
    data_transforms = {
        'train': transforms.Compose([
            # Cắt ngẫu nhiên và resize về IMG_SIZE
            transforms.RandomResizedCrop(IMG_SIZE),
            # Lật ngang ngẫu nhiên (p = 0.5)
            transforms.RandomHorizontalFlip(p=0.5),
            # Xoay ngẫu nhiên (p = 0.5)
            transforms.RandomRotation(degrees=15),
            # Thay đổi độ sáng ngẫu nhiên, độ tương phản, độ bão hòa (p = 0.5)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # Làm sắc nét ngẫu nhiên (p = 0.5)
            transforms.RandomAdjustSharpness(5.0), 
            transforms.ToTensor(),
            # Chuẩn hóa ảnh với mean và std của ImageNet
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.RandomAdjustSharpness(5.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCHSIZE, shuffle=True, num_workers=NUM_WORKERS) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names

