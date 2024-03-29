from pandas import read_csv
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch

class CSVDataset(Dataset):
    #Constructor for initially loading
    def __init__(self, path):
        df = read_csv(path, header=0)
        self.X = df.values[0:, :-1]
        self.y = df.values[0:, -1]
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

        print(self.X.shape)
        print(self.y.shape)

    # Get the number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # Get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]

    # Get the number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # Get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]

def prepare_train_dataset(df):
    train_ds = CSVDataset(df)
    train_dl = DataLoader(train_ds, batch_size=1662, shuffle=True)
    return train_dl

def prepare_test_dataset(df):
    test_ds = CSVDataset(df)
    test_dl = DataLoader(test_ds, batch_size=1024, shuffle=False)
    return test_dl
