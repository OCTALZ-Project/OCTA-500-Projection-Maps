device="cuda"
seed=42

logger=dict(
    enable=True,
    log_dir="Logs"
)

transform=dict(
    to_tensor=True,
    normalize=dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    resize=dict(
        size=(224, 224),
        antialias=True)
)

OCTADataset=dict(
    images_directory_root_path="C:/Users/omerfarukaydin/data/OCTA-500",
    labels_directory_root_path="C:/Users/omerfarukaydin/data/OCTA-500",
    num_classes=4
)

k_fold_train=dict(
    num_folds=5,
    num_epochs=100,
    batch_size=32,
    fold_indices_directory_path="C:/Users/omerfarukaydin/Desktop/OCTA/Folds"
)

model_config="configs/resnet50.py"
save_model=True
save_model_path="saved_models"

train=dict(
    model=dict(),
    optimizer=dict(
        lr=0.001),
    scheduler=dict(
        step_size=7,
        gamma=0.1
        
    )
)

