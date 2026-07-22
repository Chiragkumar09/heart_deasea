from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from data_processing import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier

df = load_data("../heart_disease_data.csv")

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

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))