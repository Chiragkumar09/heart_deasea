import pickle

from sklearn.ensemble import RandomForestClassifier

from data_processing import load_data, preprocess_data

df = load_data("heart_disease_data.csv")

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    columns_to_scale,
    categorical_columns,
    feature_columns,
) = preprocess_data(df)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

bundle = {
    "model": model,
    "scaler": scaler,
    "feature_columns": feature_columns,
    "columns_to_scale": columns_to_scale,
    "categorical_columns": categorical_columns
}

with open("heart_disease_best_model.pickle", "wb") as f:
    pickle.dump(bundle, f)

print("Model Saved")