import torch
from tqdm.auto import tqdm

def train(model, train_loader, validation_loader, scheduler, optimizer, criterion, epochs, device, train_log, validation_log):
    # Initialize progress bars for training and validation
    train_process_bar = tqdm(total=100, desc="Training", position=0, leave=True)
    val_process_bar = tqdm(total=100, desc="Validation", position=1, leave=True)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()

            epoch_train_loss += loss.item() * images.size(0)
            epoch_train_correct += correct
            
        train_process_bar.update(1)

        train_loss = epoch_train_loss / len(train_loader.dataset)
        train_accuracy = epoch_train_correct / len(train_loader.dataset)

        current_lr = optimizer.param_groups[0]['lr']

        train_process_bar.set_description(
            f"Epoch [{epoch+1}/{epochs}] train_accuracy: {train_accuracy * 100:4.1f}% train_loss: {train_loss:.4f} lr: {current_lr:.6f}"
        )

        # Log epoch sonuçlarını
        train_log.log(epoch+1, train_accuracy, train_loss, current_lr)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                correct = (preds == labels).sum().item()
                
                epoch_val_loss += loss.item() * images.size(0)
                epoch_val_correct += correct
                
                val_process_bar.update(1)

        val_loss = epoch_val_loss / len(validation_loader.dataset)
        val_accuracy = epoch_val_correct / len(validation_loader.dataset)
        
        val_process_bar.set_description(
            f"Epoch [{epoch+1}/{epochs}] validation_accuracy: {val_accuracy * 100:4.1f}% validation_loss: {val_loss:.4f} lr: {current_lr:.6f}"
        )

        # Log validation sonuçlarını
        validation_log.log(epoch+1, val_accuracy, val_loss, current_lr)

        # Step the scheduler if provided
        if scheduler:
            scheduler.step()

    train_process_bar.close()
    val_process_bar.close()
     
