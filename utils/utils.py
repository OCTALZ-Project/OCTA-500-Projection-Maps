# Standard Library
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             ConfusionMatrixDisplay, f1_score, 
                             precision_score, recall_score)
import torch
from torchvision import transforms
from datetime import datetime
from pathlib import Path

def squeeze_projection_maps(images_directory, patient_id, projection_map_headers_dir):
    projections = []
    try:
        for projection, subdir in projection_map_headers_dir.items():
            image_path = os.path.join(images_directory, subdir, f"{patient_id}.bmp")
            try:
                image = Image.open(image_path)
                projections.append(np.array(image))
            except Exception as e:
                print(f"Error loading image for patient {patient_id} and projection {projection}: {e}")
                return None
        squeezed_image = np.stack(projections, axis=-1)
        squeezed_image = torch.tensor(squeezed_image).permute(2, 0, 1)
        squeezed_image = transforms.ToPILImage()(squeezed_image)
        # squeezed_image.convert("L")
        return squeezed_image
    except Exception as e:
        print(f"Error stacking images for patient {patient_id}: {e}")
        return None
    
def load_folds(folds_folder_directory_path, class_number):
    if class_number not in [3, 4]:
        raise ValueError('Class number must be 3 or 4')
    
    fold_files_directory_path = os.path.join(folds_folder_directory_path, f'{class_number}_class_folds')
    all_files = os.listdir(fold_files_directory_path)
    
    folds = {'train_indices': [], 'val_indices': [], 'test_indices': []}
    for file in all_files:
        if file.startswith('train'):
            folds['train_indices'].append(os.path.join(fold_files_directory_path, file))
        elif file.startswith('val'):
            folds['val_indices'].append(os.path.join(fold_files_directory_path, file))
        elif file.startswith('test'):
            folds['test_indices'].append(os.path.join(fold_files_directory_path, file))

    return folds

def find_projection_paths(root_directory, ID):
    projection_paths = {}
    target_folders = ["OCTA_6mm_part8", "OCTA_3mm_part3"]
    image_extension = '.bmp'

    folders = [
        os.path.join(root_directory, folder, subfolder, "Projection Maps", "OCTA(FULL)")
        for folder, subfolder in zip(target_folders, ["OCTA_6mm", "OCTA_3mm"])
    ]

    for folder in folders:
        if os.path.exists(folder):
            for subdir, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(image_extension):
                        try:
                            key = int(''.join(filter(str.isdigit, file[:5])))
                            projection_paths[key] = os.path.join(subdir, file)
                        except ValueError:
                            continue

    return projection_paths

def find_projection_path_from_id(root_directory, patient_id):
    target_folders = ["OCTA_6mm_part8", "OCTA_3mm_part3"]
    image_extension = '.bmp'

    folders = [
        os.path.join(root_directory, folder, subfolder, "Projection Maps")
        for folder, subfolder in zip(target_folders, ["OCTA_6mm", "OCTA_3mm"])
    ]

    for folder in folders:
        if os.path.exists(folder):
            for subdir, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(image_extension) and str(patient_id) in file:
                        try:
                            return os.path.dirname(subdir)  # Return the parent directory path
                        except ValueError:
                            continue

    return None

def find_projection_directories(root_directory):
    projection_directories = {}
    target_folders = ["OCTA_6mm_part8", "OCTA_3mm_part3"]
    image_extension = '.bmp'

    folders = [
        os.path.join(root_directory, folder, subfolder, "Projection Maps")
        for folder, subfolder in zip(target_folders, ["OCTA_6mm", "OCTA_3mm"])
    ]

    for folder in folders:
        if os.path.exists(folder):
            for subdir, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(image_extension):
                        try:
                            patient_id = int(''.join(filter(str.isdigit, file[:5])))
                            projection_directories[patient_id] = os.path.dirname(subdir)
                        except ValueError:
                            continue

    return projection_directories

def find_and_combine_text_labels(root_directory):
    target_folders = ["OCTA_6mm_part8", "OCTA_3mm_part3"]
    file_name = "Text labels.xlsx"

    folders = [
        os.path.join(root_directory, folder, subfolder)
        for folder, subfolder in zip(target_folders, ["OCTA_6mm", "OCTA_3mm"])
    ]
    
    combined_df = pd.DataFrame()

    for folder in folders:
        if os.path.exists(folder):
            for subdir, _, files in os.walk(folder):
                for file in files:
                    if file == file_name:
                        file_path = os.path.join(subdir, file)
                        try:
                            df = pd.read_excel(file_path)
                            combined_df = pd.concat([combined_df, df], ignore_index=True)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
    
    return combined_df

def calculate_metrics(logs):
    all_y_true = []
    all_y_pred = []

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    for fold, log in logs.items():
        y_true = log['test_log']['y_true'][0]  # Assuming y_true and y_pred are lists of lists
        y_pred = log['test_log']['y_pred'][0]

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)

        print(f"Fold {fold}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print()

    overall_accuracy = np.mean(fold_accuracies)
    overall_precision = np.mean(fold_precisions)
    overall_recall = np.mean(fold_recalls)
    overall_f1 = np.mean(fold_f1s)

    print("Overall Metrics:")
    print(f"  Accuracy: {overall_accuracy:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1 Score: {overall_f1:.4f}")

    return {
        "fold_accuracies": fold_accuracies,
        "fold_precisions": fold_precisions,
        "fold_recalls": fold_recalls,
        "fold_f1s": fold_f1s,
        "overall_accuracy": overall_accuracy,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1
    }

def plot_training_results(log_data):
    folds = log_data.keys()
    
    for fold in folds:
        train_losses = log_data[fold]['train_log']['train_losses']
        val_losses = log_data[fold]['train_log']['val_losses']
        train_accuracies = log_data[fold]['train_log']['train_accuracies']
        val_accuracies = log_data[fold]['train_log']['val_accuracies']
        learning_rates = log_data[fold]['train_log']['learning_rates']

        epochs = range(1, len(train_losses) + 1)
        try:
            # Plotting losses
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 3, 1)
            plt.plot(epochs, train_losses, 'b', label='Training loss')
            plt.plot(epochs, val_losses, 'r', label='Validation loss')
            plt.title(f'Fold {fold} - Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
        except:
            continue
        try:
            # Plotting accuracies
            plt.subplot(1, 3, 2)
            plt.plot(epochs, train_accuracies, 'b', label='Training accuracy')
            plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
            plt.title(f'Fold {fold} - Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
        except:
            continue
        try:
            # Plotting learning rate
            plt.subplot(1, 3, 3)
            plt.plot(epochs, learning_rates, 'g', label='Learning rate')
            plt.title(f'Fold {fold} - Learning Rate')
            plt.xlabel('Epochs')
            plt.ylabel('Learning Rate')
            plt.legend()
        except:
            continue

        plt.tight_layout()
        plt.show()

def print_results(y_true, y_pred, dataset):
    print("Accuracy: {:.2f}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {:.2f}".format(precision_score(y_true, y_pred, average='macro')))
    print("Recall: {:.2f}".format(recall_score(y_true, y_pred, average='macro')))
    print("F1 Score: {:.2f}".format(f1_score(y_true, y_pred, average='macro')))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dataset.get_labels().keys()))
    disp.plot()
    plt.show()
    
def plot_all_folds_results(log_data, dataset):
    for fold in log_data.keys():
        print(f"Results for Fold {fold}:")
        
        y_true = log_data[fold]['test_log']['y_true'][0]
        y_pred = log_data[fold]['test_log']['y_pred'][0]

        print_results(y_true, y_pred, dataset)
        
def save_model(CONFIG, MODEL_CONFIG, session_dir, model, i):
    path = fr'out/{session_dir}'
    save_model_dir = Path(path) / 'saved_models'
    save_model_dir.mkdir(parents=True, exist_ok=True) 
    
    save_model_path = fr"{save_model_dir}/{str(MODEL_CONFIG.model.model)}_{CONFIG.OCTADataset.num_classes}_classes_{CONFIG.k_fold_train.num_epochs}_epoch(s)_fold_{i+1}.pth"
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")
    