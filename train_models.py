import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

from data_utils import fetch_full_dataset, build_models, prepare_uploaded_test

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    }

def main():
    X, y = fetch_full_dataset()

    test_df = pd.read_csv("test_data.csv")
    X_test, y_test = prepare_uploaded_test(test_df)

    # The bundled test file contains the first 79 public records.
    X_train = X.iloc[79:].copy()
    y_train = y.iloc[79:].copy()

    print(f"Full dataset instances: {len(X)}")
    print(f"Original feature size: {X.shape[1]}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    results = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        row = {"ML Model Name": name}
        row.update(evaluate(model, X_test, y_test))
        results.append(row)

    result_df = pd.DataFrame(results)
    print("\nModel comparison:\n")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()
