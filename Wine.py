import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    X = df.drop(columns=['quality'])
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    tree_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict(X_test)

    r2 = r2_score(y_test,y_pred)

    print(f"R² Score on Test Set: {r2}")
