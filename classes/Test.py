import torch
from classes.Config import Config

CONFIG = Config("configs/config.py")

def test(model, test_loader, device, log_test):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        log_test.log(y_true, y_pred)