import joblib
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

MODEL_PATH = "models/churn_ann.keras"

PREPROCESSOR_PATH = (
    "models/churn_ann_preprocessor.joblib"
)


# --------------------------------------------------
# 2. Load model and preprocessor
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


# --------------------------------------------------
# 3. Prediction function
# --------------------------------------------------

def predict_churn(customer_data):

    customer_df = pd.DataFrame(
        [customer_data]
    )

    customer_processed = preprocessor.transform(
        customer_df
    )

    churn_probability = float(
        model.predict(
            customer_processed,
            verbose=0
        )[0][0]
    )

    prediction = (
        1
        if churn_probability >= 0.50
        else 0
    )

    return {
        "churn_prediction": prediction,
        "churn_label": (
            "Yes"
            if prediction == 1
            else "No"
        ),
        "churn_probability": round(
            churn_probability,
            4
        )
    }


# --------------------------------------------------
# 4. Test customer
# --------------------------------------------------

if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.50,
        "TotalCharges": 966.00
    }

    result = predict_churn(
        sample_customer
    )

    print("\nANN churn prediction:")
    print(result)