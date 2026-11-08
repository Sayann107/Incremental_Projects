import os
import joblib
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ==================================================
# 1. Settings
# ==================================================

DATA_PATH = (
    "data/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

CLASSICAL_MODEL_PATH = "models/churn_v3.joblib"

ANN_MODEL_PATH = "models/churn_ann.keras"

PREPROCESSOR_PATH = (
    "models/churn_ann_preprocessor.joblib"
)

COMPARISON_PATH = (
    "models/day5_ann_comparison.csv"
)

RANDOM_STATE = 42

tf.keras.utils.set_random_seed(RANDOM_STATE)


# ==================================================
# 2. Load dataset
# ==================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DAY 5: ANN CUSTOMER CHURN MODEL")
print("=" * 70)

print("Dataset shape:", df.shape)


# ==================================================
# 3. Basic data cleaning
# ==================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


# ==================================================
# 4. Create X and y
# ==================================================

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isnull().any():
    raise ValueError(
        "Churn must contain only Yes and No."
    )


# ==================================================
# 5. Identify column types
# ==================================================

numerical_columns = (
    X.select_dtypes(include=["number"])
    .columns
    .tolist()
)

categorical_columns = (
    X.select_dtypes(
        include=["object", "category", "bool"]
    )
    .columns
    .tolist()
)

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ==================================================
# 6. Use same train-test split
# ==================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# ==================================================
# 7. Numerical preprocessing
# ==================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ==================================================
# 8. Categorical preprocessing
# ==================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


# ==================================================
# 9. Complete preprocessing
# ==================================================

preprocessor = ColumnTransformer([
    (
        "numerical",
        numerical_pipeline,
        numerical_columns
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )
])


# ==================================================
# 10. Fit only on training data
# ==================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape :",
    X_test_processed.shape
)


# ==================================================
# 11. Calculate class weights
# ==================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=y_train.unique(),
    y=y_train
)

class_weights = {
    int(class_value): float(weight)
    for class_value, weight in zip(
        y_train.unique(),
        class_weights_array
    )
}

print("\nClass weights:")
print(class_weights)


# ==================================================
# 12. Build ANN
# ==================================================

number_of_features = X_train_processed.shape[1]

ann_model = Sequential([
    Input(shape=(number_of_features,)),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


# ==================================================
# 13. Compile ANN
# ==================================================

ann_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)

print("\nANN architecture:")
ann_model.summary()


# ==================================================
# 14. Early stopping
# ==================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


# ==================================================
# 15. Train ANN
# ==================================================

history = ann_model.fit(
    X_train_processed,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)


# ==================================================
# 16. ANN probabilities and predictions
# ==================================================

ann_probabilities = (
    ann_model.predict(
        X_test_processed,
        verbose=0
    )
    .reshape(-1)
)

ann_predictions = (
    ann_probabilities >= 0.50
).astype(int)


# ==================================================
# 17. ANN metrics
# ==================================================

ann_accuracy = accuracy_score(
    y_test,
    ann_predictions
)

ann_precision = precision_score(
    y_test,
    ann_predictions,
    zero_division=0
)

ann_recall = recall_score(
    y_test,
    ann_predictions,
    zero_division=0
)

ann_f1 = f1_score(
    y_test,
    ann_predictions,
    zero_division=0
)

ann_roc_auc = roc_auc_score(
    y_test,
    ann_probabilities
)

print("\n" + "=" * 70)
print("ANN HOLD-OUT TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {ann_accuracy:.4f}")
print(f"Precision: {ann_precision:.4f}")
print(f"Recall   : {ann_recall:.4f}")
print(f"F1-score : {ann_f1:.4f}")
print(f"ROC-AUC  : {ann_roc_auc:.4f}")

print("\nANN classification report:")

print(
    classification_report(
        y_test,
        ann_predictions,
        target_names=[
            "No Churn",
            "Churn"
        ],
        zero_division=0
    )
)


# ==================================================
# 18. Load classical champion
# ==================================================

if not os.path.exists(CLASSICAL_MODEL_PATH):
    raise FileNotFoundError(
        "models/churn_v3.joblib was not found. "
        "Run Day 3 tune_models.py first."
    )

classical_model = joblib.load(
    CLASSICAL_MODEL_PATH
)


# ==================================================
# 19. Classical model predictions
# ==================================================

classical_predictions = classical_model.predict(
    X_test
)

classical_probabilities = (
    classical_model.predict_proba(
        X_test
    )[:, 1]
)


# ==================================================
# 20. Classical model metrics
# ==================================================

classical_accuracy = accuracy_score(
    y_test,
    classical_predictions
)

classical_precision = precision_score(
    y_test,
    classical_predictions,
    zero_division=0
)

classical_recall = recall_score(
    y_test,
    classical_predictions,
    zero_division=0
)

classical_f1 = f1_score(
    y_test,
    classical_predictions,
    zero_division=0
)

classical_roc_auc = roc_auc_score(
    y_test,
    classical_probabilities
)


# ==================================================
# 21. Create comparison
# ==================================================

comparison = pd.DataFrame([
    {
        "Model": "Day 3 Classical Champion",
        "Accuracy": round(
            classical_accuracy,
            4
        ),
        "Precision": round(
            classical_precision,
            4
        ),
        "Recall": round(
            classical_recall,
            4
        ),
        "F1": round(
            classical_f1,
            4
        ),
        "ROC-AUC": round(
            classical_roc_auc,
            4
        )
    },
    {
        "Model": "Day 5 ANN",
        "Accuracy": round(
            ann_accuracy,
            4
        ),
        "Precision": round(
            ann_precision,
            4
        ),
        "Recall": round(
            ann_recall,
            4
        ),
        "F1": round(
            ann_f1,
            4
        ),
        "ROC-AUC": round(
            ann_roc_auc,
            4
        )
    }
])


# ==================================================
# 22. Select winner
# ==================================================

comparison = comparison.sort_values(
    by=[
        "Recall",
        "ROC-AUC",
        "F1"
    ],
    ascending=[
        False,
        False,
        False
    ]
).reset_index(drop=True)

print("\n" + "=" * 70)
print("ANN VS CLASSICAL ML")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)

winner = comparison.iloc[0]["Model"]

print("\nDay 5 winner:", winner)


# ==================================================
# 23. Save ANN and preprocessing
# ==================================================

os.makedirs(
    "models",
    exist_ok=True
)

ann_model.save(
    ANN_MODEL_PATH
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

comparison.to_csv(
    COMPARISON_PATH,
    index=False
)


# ==================================================
# 24. Save training history
# ==================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    "models/day5_ann_training_history.csv",
    index=False
)


# ==================================================
# 25. Final output
# ==================================================

print("\nFiles saved:")

print(
    "ANN model            :",
    ANN_MODEL_PATH
)

print(
    "ANN preprocessor     :",
    PREPROCESSOR_PATH
)

print(
    "Model comparison     :",
    COMPARISON_PATH
)

print(
    "ANN training history :",
    "models/day5_ann_training_history.csv"
)