import os
import pandas as pd
from classes.OCTADataset import OCTADataset


class Folds:
    def __init__(self, dataset: OCTADataset, folds_folder_directory_path: str, class_number: int = 4):
        """
        Manages loading and organizing data folds for training, validation, and testing.

        Args:
            dataset (OCTADataset): The dataset object containing patient data.
            random (bool): Whether to use random folds (not implemented yet).
            folds_folder_directory_path (str): Path to the directory containing fold information.
            class_number (int, optional): The number of classes in the dataset (3 or 4). Defaults to 4.
        """
        # if not random:
        if class_number not in [3, 4]:
            raise ValueError('Class number must be 3 or 4')
        # self.random = random  # Not used currently
        self.folds_folder_directory_path = folds_folder_directory_path
        self.class_number = class_number
        self.dataset = dataset
        self.folds_dict = self.create_folds_dict()

    def _load_fold_indices(self, fold_df: pd.DataFrame) -> list:
        """
        Loads indices from a fold DataFrame based on patient IDs.

        Args:
            fold_df (pd.DataFrame): DataFrame containing patient IDs for a specific fold.

        Returns:
            list: List of indices corresponding to the patients in the fold.
        """
        return [self.dataset.patient_id_to_index[pid] 
                for pid in fold_df['ID_nid'].values.tolist() 
                if pid in self.dataset.patient_id_to_index]

    def create_folds_dict(self) -> dict:
        """
        Creates a dictionary containing train, validation, and test indices for each fold.

        Returns:
            dict: Dictionary of folds, where each fold contains train, val, and test indices.
        """
        fold_files_directory_path = os.path.join(self.folds_folder_directory_path, f'{self.class_number}_class_folds')
        fold_files = os.listdir(fold_files_directory_path)

        folds_dict = {}
        for i in range(len(fold_files) // 3):  # Assuming consistent naming convention
            train_df = pd.read_csv(os.path.join(fold_files_directory_path, f'train_fold_{i}.csv'))
            val_df = pd.read_csv(os.path.join(fold_files_directory_path, f'val_fold_{i}.csv'))
            test_df = pd.read_csv(os.path.join(fold_files_directory_path, f'test_fold_{i}.csv'))

            folds_dict[f'fold_{i}'] = {  # Adjust index to start from 0
                'train': self._load_fold_indices(train_df),
                'val': self._load_fold_indices(val_df),
                'test': self._load_fold_indices(test_df)
            }

        return folds_dict

    def get_fold(self, fold_index: int) -> dict:
        """
        Retrieves the train, validation, and test indices for a specific fold.

        Args:
            fold_index (int): The index of the desired fold.

        Returns:
            dict: Dictionary containing train, val, and test indices for the specified fold.
        """
        if f'fold_{fold_index}' in self.folds_dict:
            return self.folds_dict[f'fold_{fold_index}']
        else:
            raise ValueError(f"Fold index {fold_index} not found.")
        
    def num_folds(self) -> int:
        """
        Returns the number of folds in the dataset.

        Returns:
            int: The number of folds.
        """
        return len(self.folds_dict)
    
    def folds_dict(self) -> dict:
        """
        Returns the dictionary of folds.

        Returns:
            dict: Dictionary of folds.
        """
        return self._folds_dict