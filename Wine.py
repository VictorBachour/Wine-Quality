import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class Model(nn.Module):
    def __init__(self, in_features=11, h1=8, h2=9, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x




if __name__ == "__main__":
    model = Model()

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    X = df.drop(columns=['quality'])
    y = df['quality']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=.2, random_state=42)


    loss = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=.01)

    best_loss = float()