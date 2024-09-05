import csv
from pathlib import Path

class Logger:
    def __init__(self, session_dir):
        path = fr'out/{session_dir}'
        self.log_dir = Path(path) / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def create_fold_dir(self, fold_number):
        fold_dir = self.log_dir / f'fold_{fold_number}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        return fold_dir

class LogFile:
    def __init__(self, fold_dir, log_type, headers):
        self.fold_dir = fold_dir
        self.log_type = log_type
        self.file_path = self.create_log_file(headers)
        
    def create_log_file(self, headers):
        file_name = f'{self.log_type}.csv'
        log_file_path = self.fold_dir / file_name
        with log_file_path.open(mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        return log_file_path
        
    def log(self, row):
        with self.file_path.open(mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

class TrainLogger(LogFile):
    def __init__(self, fold_dir):
        headers = ['Epoch', 'Accuracy', 'Loss', 'Learning Rate']
        super().__init__(fold_dir, 'train', headers)
        
    def log(self, epoch, accuracy, loss, lr):
        super().log([epoch, accuracy, loss, lr])

class ValidationLogger(LogFile):
    def __init__(self, fold_dir):
        headers = ['Epoch', 'Accuracy', 'Loss', 'Learning Rate']
        super().__init__(fold_dir, 'validation', headers)
        
    def log(self, epoch, accuracy, loss, lr):
        super().log([epoch, accuracy, loss, lr])
    
class TestLogger(LogFile):
    def __init__(self, fold_dir):
        headers = ['y_pred', 'y_true']
        super().__init__(fold_dir, 'test', headers)
        
    def log(self, y_true, y_pred):
        super().log([y_pred, y_true])