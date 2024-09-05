import torch
import torch.nn as nn
from torchvision.models import resnet50
from enums.model import Model
import timm

def initialize_model(CONFIG):
    if CONFIG.model.model == Model.RESNET50:
        model = resnet50(weights=CONFIG.model.weights)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, CONFIG.model.num_classes)
        
    elif CONFIG.model.model == Model.ViT:
        model = timm.create_model(CONFIG.model.weights, pretrained=CONFIG.model.pretrained, num_classes=CONFIG.model.num_classes)
        model.head = torch.nn.Linear(in_features=CONFIG.model.in_features, out_features=CONFIG.model.num_classes, bias=CONFIG.model.bias)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model
