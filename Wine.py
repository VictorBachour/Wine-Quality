import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
class Model(nn.Module):
    def __init__(self, in_features=11, h1=16, h2=9, out_features=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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


    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(100):
        model.train()

        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/ {100}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

        y_pred = y_pred.numpy()
        y_test = y_test.numpy()

        r2 = r2_score(y_test, y_pred)
        print(f"R² Score on Test Set: {r2}")
