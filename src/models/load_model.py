import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from torchvision import models
from utils.load_config import load_config

cudnn.benchmark = True

def load_model():
    # load configuration
    config = load_config('config.yaml')
    MODEL_NAME = config['MODEL']['MODEL_NAME'] if config['MODEL']['MODEL_NAME'] else 'resnet50'
    CHECKPOINT = config['MODEL']['CHECKPOINT'] if config['MODEL']['CHECKPOINT'] else ''
    NUMCLASS = config['MODEL']['NUMCLASS'] if config['MODEL']['NUMCLASS'] else 2
    
    # Resolve checkpoint path from project root
    if CHECKPOINT:
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        checkpoint_path = PROJECT_ROOT / CHECKPOINT.lstrip('../')
        CHECKPOINT = str(checkpoint_path)

    try:
        # Resnet
        if MODEL_NAME == "resnet18":            
            if CHECKPOINT:
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet18(weights='DEFAULT')
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
            # freeze all layers
            for param in model.parameters():
                    param.requires_grad = False

            # open last layer of feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet34":            
            if CHECKPOINT:
                model = models.resnet34(weights=None)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet34(weights='DEFAULT')
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
            # freeze all layers
            for param in model.parameters():
                    param.requires_grad = False
                    
            # open last layer of feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet50":            
            if CHECKPOINT:
                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet50(weights='DEFAULT')
                # print(model)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
            # freeze all layers
            for param in model.parameters():
                    param.requires_grad = False
                    
            # open last layer of feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet101":            
            if CHECKPOINT:
                model = models.resnet101(weights=None)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet101(weights='DEFAULT')
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
            # freeze all layers
            for param in model.parameters():
                    param.requires_grad = False
                    
            # open last layer of feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True
        
        elif MODEL_NAME == "resnet152":            
            if CHECKPOINT:
                model = models.resnet152(weights=None)
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)

                if not torch.cuda.is_available():
                    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu')) 
                else: 
                    checkpoint = torch.load(CHECKPOINT)
                    
                model.load_state_dict(checkpoint['model_state_dict'])
                
            else: 
                model = models.resnet152(weights='DEFAULT')
                model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
            # freeze all layers
            for param in model.parameters():
                    param.requires_grad = False
                    
            # open last layer of feature
            model.layer4.requires_grad_(True)
            model.avgpool.requires_grad_(True)
            for param in model.fc.parameters():
                param.requires_grad = True

        return model
    except:
        print('Error: Could not load model.')
        exit(1)