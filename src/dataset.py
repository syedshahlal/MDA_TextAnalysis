from pandas import read_csv
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

class CSVDataset(Dataset):
    """Dataset class for loading data from CSV files."""
    def __init__(self, path):
        df = read_csv(path, header=0)
        # Assume last column is target
        self.X = df.iloc[:, :-1].values.astype('float32')
        self.y = df.iloc[:, -1].values.astype('float32')
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

def prepare_train_dataset(path):
    train = CSVDataset(path)
    train_dl = DataLoader(train, batch_size=1662, shuffle=True)
    return train_dl

def prepare_test_dataset(path):
    test = CSVDataset(path)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return test_dl