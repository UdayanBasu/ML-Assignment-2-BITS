import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

from data_utils import (
    fetch_full_dataset, build_models, prepare_uploaded_test,
    FEATURE_COLUMNS, TARGET
)

st.set_page_config(
    page_title="Online Shopper Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Online Shopper Purchase Intention Classification")
st.caption(
    "UCI Online Shoppers Purchasing Intention dataset — binary classification."
)

@st.cache_resource
def train_all_models():
    X, y = fetch_full_dataset()

    # The bundled CSV uses the first records from the public dataset.
    # Exclude the first 79 rows so the uploaded file is not used for training.
    X_train = X.iloc[79:].copy()
    y_train = y.iloc[79:].copy()

    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)

    return models

def metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

with st.sidebar:
    st.header("Controls")
    selected_model = st.selectbox(
        "Select model",
        [
            "Logistic Regression",
            "Decision Tree",
            "kNN",
            "Naive Bayes",
            "Random Forest",
        ]
    )
    st.info("Upload the included test_data.csv file.")

models = train_all_models()

st.subheader("1. Upload Test Data")
uploaded = st.file_uploader("Choose test_data.csv", type=["csv"])

if uploaded is None:
    st.info("Please upload the included `test_data.csv`.")
    st.stop()

df = pd.read_csv(uploaded)

try:
    X_test, y_test = prepare_uploaded_test(df)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded {len(df)} test records.")
with st.expander("Preview uploaded data"):
    st.dataframe(df.head(10), use_container_width=True)

model = models[selected_model]
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

result = metrics(y_test, y_pred, y_prob)

st.subheader(f"2. Metrics — {selected_model}")
cols = st.columns(6)
for col, (name, value) in zip(cols, result.items()):
    col.metric(name, f"{value:.4f}")

st.subheader("3. Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 4))
img = ax.imshow(cm)
ax.set_title(f"Confusion Matrix — {selected_model}")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1], labels=["No Purchase", "Purchase"])
ax.set_yticks([0, 1], labels=["No Purchase", "Purchase"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(img, ax=ax)
st.pyplot(fig)

st.subheader("4. Classification Report")
report = classification_report(
    y_test, y_pred,
    target_names=["No Purchase", "Purchase"],
    output_dict=True,
    zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

st.subheader("5. Model Comparison")
rows = []
for name, current_model in models.items():
    pred = current_model.predict(X_test)
    prob = current_model.predict_proba(X_test)[:, 1]
    row = {"ML Model Name": name}
    row.update(metrics(y_test, pred, prob))
    rows.append(row)

comparison = pd.DataFrame(rows)
st.dataframe(
    comparison.style.format({
        "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}"
    }),
    use_container_width=True
)

winner = comparison.sort_values(
    ["F1", "MCC", "AUC"], ascending=False
).iloc[0]

st.success(
    f"Overall winner on the uploaded test file: **{winner['ML Model Name']}**"
)

st.caption(
    "Target: Revenue. FALSE/0 = no purchase, TRUE/1 = purchase."
)
