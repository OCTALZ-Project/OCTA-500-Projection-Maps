from enums.model import Model

model=dict(
    model=Model.ViT,
    weights='vit_large_patch16_224',
    pretrained=True,
    bias=True,
    in_features=1024,
    num_classes=4
)