
import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="centered"
)

# Sample educational dataset
data = {
    "income": [25000, 40000, 50000, 20000, 60000, 35000, 80000, 30000],
    "loan_amount": [10000, 15000, 20000, 12000, 25000, 10000, 30000, 18000],
    "credit_score": [600, 700, 750, 550, 800, 650, 780, 580],
    "approved": [1, 1, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[["income", "loan_amount", "credit_score"]]
y = df["approved"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)
model.fit(X, y)

st.title("💳 Credit Scoring System")
st.write("Machine Learning Based Credit Approval Prediction")

st.info("Enter applicant details below to generate a demo prediction.")

income = st.number_input(
    "Annual Income (₹)",
    min_value=10000,
    max_value=10000000,
    value=50000,
    step=5000
)

loan = st.number_input(
    "Loan Amount (₹)",
    min_value=1000,
    max_value=5000000,
    value=15000,
    step=1000
)

score = st.slider(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=700
)

if st.button("Predict Credit Approval", type="primary"):
    applicant = pd.DataFrame({
        "income": [income],
        "loan_amount": [loan],
        "credit_score": [score]
    })

    prediction = model.predict(applicant)[0]
    probability = model.predict_proba(applicant)[0][1]

    if prediction == 1:
        st.success("Demo Prediction: Credit Approved")
    else:
        st.error("Demo Prediction: Credit Not Approved")

    st.metric("Estimated Approval Probability", f"{probability * 100:.2f}%")

    st.caption(
        "Educational demonstration only. Model uses 8 sample records "
        "and is not suitable for real lending decisions."
    )

st.divider()
st.caption("Developed by Ashwini Shelke | Credit Scoring ML Project")
