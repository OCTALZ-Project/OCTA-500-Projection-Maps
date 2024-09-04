from torchvision.models import ResNet50_Weights
from enums.model import Model

model=dict(
    model=Model.RESNET50,
    weights=ResNet50_Weights.IMAGENET1K_V1,
    num_classes=4
)