import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from pathlib import Path
from torchvision import models
from utils.load_config import load_config

cudnn.benchmark = True

def _resolve_checkpoint_path(checkpoint: str) -> str:
    if not checkpoint:
        return ""
    path = Path(checkpoint)
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parent.parent.parent
    if checkpoint.startswith(".."):
        return str((project_root / checkpoint.lstrip("../")).resolve())
    return str((project_root / checkpoint).resolve())


def _load_checkpoint(model: nn.Module, checkpoint_path: str, model_name: str, numclass: int) -> None:
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Some checkpoints are just the raw state_dict.
            state_dict = checkpoint

        model_state = model.state_dict()
        filtered_state = dict(state_dict)
        mismatched_head = []
        for key in ("fc.weight", "fc.bias"):
            if key in filtered_state and key in model_state:
                if filtered_state[key].shape != model_state[key].shape:
                    mismatched_head.append(key)
                    del filtered_state[key]

        if mismatched_head:
            model.load_state_dict(filtered_state, strict=False)
        else:
            model.load_state_dict(filtered_state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint incompatible with model '{model_name}' and NUMCLASS={numclass}."
        ) from exc


def load_model():
    # load configuration
    config = load_config('config.yaml')
    if not config:
        raise ValueError("Failed to load config.yaml")
    MODEL_NAME = config['MODEL']['MODEL_NAME'] if config['MODEL']['MODEL_NAME'] else 'resnet50'
    CHECKPOINT = config['MODEL']['CHECKPOINT'] if config['MODEL']['CHECKPOINT'] else ''
    NUMCLASS = config['MODEL']['NUMCLASS'] if config['MODEL']['NUMCLASS'] else 2
    PRETRAINED = config['MODEL'].get('PRETRAINED', True)
    
    CHECKPOINT = _resolve_checkpoint_path(CHECKPOINT)
    if CHECKPOINT and not Path(CHECKPOINT).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    # Resnet
    if MODEL_NAME == "resnet18":            
        if CHECKPOINT:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
            _load_checkpoint(model, CHECKPOINT, MODEL_NAME, NUMCLASS)
        else: 
            model = models.resnet18(weights='DEFAULT' if PRETRAINED else None)
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
            _load_checkpoint(model, CHECKPOINT, MODEL_NAME, NUMCLASS)
        else: 
            model = models.resnet34(weights='DEFAULT' if PRETRAINED else None)
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
            _load_checkpoint(model, CHECKPOINT, MODEL_NAME, NUMCLASS)
        else: 
            model = models.resnet50(weights='DEFAULT' if PRETRAINED else None)
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
            _load_checkpoint(model, CHECKPOINT, MODEL_NAME, NUMCLASS)
        else: 
            model = models.resnet101(weights='DEFAULT' if PRETRAINED else None)
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
            _load_checkpoint(model, CHECKPOINT, MODEL_NAME, NUMCLASS)
        else: 
            model = models.resnet152(weights='DEFAULT' if PRETRAINED else None)
            model.fc = nn.Linear(model.fc.in_features, NUMCLASS)
        # freeze all layers
        for param in model.parameters():
                param.requires_grad = False
                
        # open last layer of feature
        model.layer4.requires_grad_(True)
        model.avgpool.requires_grad_(True)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

    return model
