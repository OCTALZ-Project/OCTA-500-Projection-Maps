from torch.utils.data import Dataset

from utils.utils import find_projection_directories, find_and_combine_text_labels, squeeze_projection_maps

class OCTADataset(Dataset):
    def __init__(self, images_directory_root_path, labels_df=None, labels_directory_root_path=None, transform=None, num_classes=[3,4,7]):
        self.images_directory = images_directory_root_path
        self.labels_directory = labels_directory_root_path
        self.projection_directories_dictionary = find_projection_directories(images_directory_root_path)
        if labels_directory_root_path is not None and labels_df is None:
            self.labels_df = find_and_combine_text_labels(labels_directory_root_path)
        elif labels_df is not None:
            self.labels_df = labels_df
        else:
            raise ValueError("labels_df or labels_directory_root_path must be provided")
        self.transform = transform
        self.num_classes = num_classes
        self.projection_map_headers_dictionary = {
            "FULL": "OCTA(FULL)",
            "ILM_OPL": "OCTA(ILM_OPL)",
            "OPL_BM": "OCTA(OPL_BM)"
        }
        self.original_labels = {'NORMAL': 0, 'AMD': 1, 'DR': 2, 'CNV': 3, 'CSC': 4, 'RVO': 5, 'OTHERS': 6}
        
        if num_classes == 7:
            self.label_map = self.original_labels
        elif num_classes == 3:
            self.label_map = {'NORMAL': 0, 'AMD': 1, 'DR': 2}
        elif num_classes == 4:
            self.label_map = {'NORMAL': 0, 'AMD': 1, 'DR': 2, 'OTHERS': 3}
        else:
            raise ValueError("num_classes must be 3, 4, or 7")

        # Create a patient ID to index mapping
        self.patient_id_to_index = {int(row['ID']): idx for idx, row in self.labels_df.iterrows()}

    def get_labels(self):
        return self.label_map
    
    def get_labels_df(self):
        return self.labels_df

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        patient_id = int(row['ID'])
        label = row['Disease']

        if self.num_classes == 3 or self.num_classes == 4:
            if label not in ['NORMAL', 'AMD', 'DR']:
                label = 'OTHERS'
                
        label = self.label_map[label]
        projection_path = self.projection_directories_dictionary[patient_id]
        squeezed_image = squeeze_projection_maps(projection_path, patient_id, self.projection_map_headers_dictionary)
        
        if self.transform:
            squeezed_image = self.transform(squeezed_image)

        return squeezed_image, label