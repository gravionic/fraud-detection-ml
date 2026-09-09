# 1. Import required libraries

from pathlib import Path
import streamlit as st
import joblib as  jl
import pandas as pd
import numpy as np
import shap
from groq import Groq
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv() #environment variables from .env file

# 2. Path settings

Basedir = Path(__file__).resolve().parent.parent
MODEL_DIR = Basedir / "model"



# 3. Load the trained pipeline and initialize the Groq client

model = jl.load(MODEL_DIR / "fraud_detection_model.pkl")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 4. Page settings

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")
st.title("Fraud Detection")
st.markdown("##### Predict whether the transaction is fraudulent using a trained ML model and explain each prediction with SHAP Explainable AI.")
st.markdown("---")


# 5. Define feature labels for SHAP explanations

FEATURE_LABELS = {
    "num__amount": "Transaction amount",
    "num__oldbalanceOrg": "Origin account previous balance",
    "num__newbalanceOrig": "Origin account new balance",
    "num__OrigBalanceChange": "Origin account balance change",

    "num__oldbalanceDest": "Destination account previous balance",
    "num__newbalanceDest": "Destination account new balance",
    "num__DestBalanceChange": "Destination account balance change",

    "cat__type_PAYMENT": "Payment transaction type",
    "cat__type_CASH_OUT": "Cash-out transaction type",
    "cat__type_TRANSFER": "Transfer transaction type",
    "cat__type_CASH_IN": "Cash-in transaction type",
    "cat__type_DEBIT": "Debit transaction type"
}

# 6. Function to generate AI analysis using Groq API

def generate_ai_analysis(prediction, probability, transaction_type, top_features): 

    prompt = f"""
You are an AI assistant helping a fraud analyst review a financial transaction.

The XGBoost model has already made the fraud prediction.
You must NOT change or question that prediction.

Your job is to explain the model's decision using the SHAP factors below.

Transaction type: {transaction_type}
Model prediction: {"Fraudulent" if prediction == 1 else "Legitimate"}
Fraud probability: {probability:.1%}


Top SHAP factors:
{top_features}

Write the response using exactly these sections:

Why this result?
Write 3-4 clear sentences explaining why the model reached this result.
Explain how the strongest SHAP factors increased or decreased the model's
fraud risk.

Key Drivers
- Explain the most important SHAP factor in 1 clear and concise sentence.
- Explain the second most important SHAP factor in 1 clear and concise sentence.
- Explain the third most important SHAP factor in 1 clear and concise sentence.

For the recommended action:
- If fraud probability is above 70%, recommend a strong action such as
  holding the transaction and investigating it.
- If fraud probability is between 40% and 70%, recommend manual review
  and/or monitoring.
- If fraud probability is below 40%, recommend proceeding and transaction looking safe.
- Keep the recommendation practical and concise.



Rules:
- Vary the sentence structure and don't repeat the same phrasing for each sentences or bullet points.
- Do not change the model prediction.
- Only use the information provided.
- Do not invent account history or other transaction information.
- Explain SHAP factors as influences on the model's prediction.
- Keep the language simple and professional.


Recommended Action
[one concise practical recommendation]
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=700
    )

    return response.choices[0].message.content


# 7. User input

type = st.selectbox("Transaction Type", ["PAYMENT", "CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0, step=0.001, format="%.3f")
oldbalanceOrg=st.number_input("Old Balance of Origin Account", min_value=0.0, value=1000.0, step=0.001, format="%.3f")
newbalanceOrig=st.number_input("New Balance of Origin Account", min_value=0.0, value=0.0, step=0.001, format="%.3f")
oldbalanceDest=st.number_input("Old Balance of Destination Account", min_value=0.0, value=0.0, step=0.001, format="%.3f")
newbalanceDest=st.number_input("New Balance of Destination Account", min_value=0.0, value=0.0, step=0.001, format="%.3f")    

OrigBalanceChange= oldbalanceOrg - newbalanceOrig
DestBalanceChange= newbalanceDest - oldbalanceDest

# 8. Predict section

predict = st.button("Predict Fraud")

if predict:

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        "type": [type],
        "amount": [amount],
        "newbalanceOrig": [newbalanceOrig],
        "oldbalanceOrg": [oldbalanceOrg],
        "OrigBalanceChange": [OrigBalanceChange],
        "DestBalanceChange": [DestBalanceChange],
        "oldbalanceDest": [oldbalanceDest],
        "newbalanceDest": [newbalanceDest]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    confidence = abs(probability-0.5)*2
    st.markdown("---")

    # 9. Display transaction details and prediction results

    st.subheader("Transaction Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Type", type)

    with col2:
        st.metric("Amount", f"${amount:,.0f}")

    with col3:
        st.metric("Origin Balance Change", f"${OrigBalanceChange:,.0f}")

    with col4:
        st.metric("Destination Balance Change", f"${DestBalanceChange:,.0f}")


    st.markdown("---")
    
    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{probability*100:.1f}%"
        )

    with col2:

        if probability < 0.4:
            risk = "Low"

        elif probability < 0.7:
            risk = "Moderate"

        else:
            risk = "High"

        st.metric(
            "Risk Level",
            risk
        )

    with col3:

        st.metric(
            "Prediction Confidence",
            f"{confidence*100:.1f}%"
        )

    # Display the prediction result
    if prediction == 1:
        st.error(f"The transaction is predicted to be fraudulent.")
    else:
        st.success(f"The transaction is predicted to be legitimate.")    
        
    st.markdown("---")

    # 10. SHAP Explanation

    st.subheader("Local SHAP Explanation")

    # Separate preprocessing from the XGBoost model
    preprocessor = model[:-1]
    xgb_model = model.steps[-1][1]

    # Transform the transaction using the same preprocessing
    input_transformed = preprocessor.transform(input_data)

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # SHAP explainer for the actual XGBoost model
    explainer = shap.TreeExplainer(xgb_model)

    # Explain this transaction
    explanation = explainer(input_transformed)

    # Add feature names
    explanation.feature_names = feature_names

    # Get local SHAP values for this transaction
    shap_values = explanation.values

    if shap_values.ndim == 3:
        local_shap_values = shap_values[0, :, 1]
    else:
        local_shap_values = shap_values[0]

    # Match each feature with its SHAP value
    feature_contributions = list(
        zip(feature_names, local_shap_values)
    )

    # Sort by strongest influence
    feature_contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # Get top 3 drivers
    top_features = []

    for feature, value in feature_contributions[:3]:

        friendly_name = FEATURE_LABELS.get(
            feature,
            feature
        )

        direction = (
            "increased"
            if value > 0
            else "decreased"
        )

        top_features.append(
            f"{friendly_name} {direction} "
            f"the model's fraud risk "
            f"(SHAP value: {value:+.2f})"
        )

    top_features_text = "\n".join(f"- {feature}" for feature in top_features) # List of top feature contributions

    # 11. AI analysis
    
    try:
        ai_analysis = generate_ai_analysis(prediction, probability, type, top_features_text)
        
    except Exception as e:
        
        st.error(f"AI analysis error: {e}")
        
        ai_analysis = ("AI analysis is currently unavailable. Please try again later.")

               
    # 12.  SHAP waterfall plot

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[0], max_display=10, show=False )
    left, right = st.columns([1.6,1])
    
    with left:
        st.pyplot(plt.gcf(), use_container_width=True)
    with right:    
        # Display key drivers in the right column

        st.write(ai_analysis)
        plt.close()

              