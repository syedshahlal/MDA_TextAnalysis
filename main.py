import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sigmoid
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class FraudDataset(Dataset):
    """
    A dataset class for fraud detection data.
    """
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32').reshape((-1, 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

class FraudDetectionModel(Module):
    """
    A neural network for fraud detection.
    """
    def __init__(self, n_inputs):
        super(FraudDetectionModel, self).__init__()
        self.hidden1 = Linear(n_inputs, 57)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()

        self.hidden2 = Linear(57, 22)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()

        self.hidden3 = Linear(22, 107)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()

        self.hidden4 = Linear(107, 202)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = Sigmoid()

        self.hidden5 = Linear(202, 162)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = Sigmoid()

        self.hidden6 = Linear(162, 1)
        xavier_uniform_(self.hidden6.weight)
        self.act6 = Sigmoid()

    def forward(self, X):
        X = self.act1(self.hidden1(X))
        X = self.act2(self.hidden2(X))
        X = self.act3(self.hidden3(X))
        X = self.act4(self.hidden4(X))
        X = self.act5(self.hidden5(X))
        X = self.act6(self.hidden6(X))
        return X

class FraudDetectionSystem:
    """
    A system for fraud detection including data preparation, model training, and evaluation.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude non-numeric columns from scaling
        columns_to_scale = [col for col in numeric_cols if col not in ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
                       'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk',
                       'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets',
                       'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf', 'issue']]

        df[columns_to_scale] = self.scaler.fit_transform(df[columns_to_scale])

        return df

    def split_data(self, df, cutoff_year):
        train_df = df[df['fyear'] <= cutoff_year]
        test_df = df[df['fyear'] > cutoff_year]

        return train_df, test_df

    def prepare_datasets(self, train_df, test_df, features, target):
        X_train, y_train = train_df[features].values, train_df[target].values
        X_test, y_test = test_df[features].values, test_df[target].values

        train_dataset = FraudDataset(X_train, y_train)
        test_dataset = FraudDataset(X_test, y_test)

        return DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(test_dataset, batch_size=32, shuffle=False)

    def train_model(self, train_loader, input_features, epochs=10):
        self.model = FraudDetectionModel(input_features)
        optimizer = SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.BCELoss()

        for epoch in range(epochs):
            self.model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def evaluate_model(self, test_loader):
        self.model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                outputs = self.model(inputs)
                predictions.append(outputs.numpy())
                actuals.append(targets.numpy())
        predictions, actuals = np.vstack(predictions), np.vstack(actuals)
        auc_score = roc_auc_score(actuals, predictions)
        print(f"AUC Score: {auc_score}")

def main():
    # Assume 'data_path' is defined or replace with actual CSV file path
    data_path = "https://media.githubusercontent.com/media/syedshahlal/MDA_TextAnalysis/main/dataset/merged_compustat_and_labels.csv"
    system = FraudDetectionSystem(data_path)
    df = system.load_and_preprocess_data()
    train_df, test_df = system.split_data(df, cutoff_year=2002)
    features = [col for col in df.columns if col not in ['act', 'ap', 'at', 'ceq', 'che', 'cogs', 'csho', 'dlc', 'dltis', 'dltt', 'dp', 'ib',
                       'invt', 'ivao', 'ivst', 'lct', 'lt', 'ni', 'ppegt', 'pstk', 're', 'rect', 'sale', 'sstk',
                       'txp', 'txt', 'xint', 'prcc_f', 'dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'soft_assets',
                       'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'dpi', 'reoa', 'EBIT', 'ch_fcf', 'issue']]
    train_loader, test_loader = system.prepare_datasets(train_df, test_df, features, 'misstate')
    system.train_model(train_loader, len(features))
    system.evaluate_model(test_loader)

if __name__ == "__main__":
    main()
