import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):

    X = df.drop("target", axis=1)
    y = df["target"]

    columns_to_scale = [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]

    categorical_columns = [
        "cp",
        "restecg",
        "slope",
        "ca",
        "thal"
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True,
        dtype=int
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()

    X_train[columns_to_scale] = scaler.fit_transform(
        X_train[columns_to_scale]
    )

    X_test[columns_to_scale] = scaler.transform(
        X_test[columns_to_scale]
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        columns_to_scale,
        categorical_columns,
        X.columns.tolist()
    )