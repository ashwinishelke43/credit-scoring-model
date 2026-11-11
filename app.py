

import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CrediSense | Credit Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================================================
# 2. CUSTOM DESIGN
# ==================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081426, #10233d);
    color: #f4f7fb;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(110deg, #17385b, #126b79);
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
    border: 1px solid #315b78;
}

.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #d6e9f3;
}

.panel {
    background: #142942;
    padding: 23px;
    border: 1px solid #294561;
    border-radius: 17px;
    margin-bottom: 18px;
}

.result-approved {
    background: linear-gradient(120deg, #104b42, #14665b);
    border: 1px solid #36b99a;
    padding: 25px;
    border-radius: 17px;
}

.result-rejected {
    background: linear-gradient(120deg, #542c3b, #713447);
    border: 1px solid #e17b89;
    padding: 25px;
    border-radius: 17px;
}

.small-label {
    color: #a9c0d5;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.big-value {
    color: white;
    font-size: 27px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #91a8bd;
    padding: 25px;
    font-size: 13px;
    border-top: 1px solid #294561;
    margin-top: 35px;
}

div[data-testid="stMetric"] {
    background: #142942;
    border: 1px solid #294561;
    padding: 15px;
    border-radius: 14px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #16a6a0, #237bd1);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    min-height: 45px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid white;
    color: white;
}

label {
    color: #e1eaf4 !important;
}

[data-testid="stCaptionContainer"] {
    color: #a9c0d5;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. SAMPLE EDUCATIONAL DATASET
# ==================================================

data = {
    "income": [
        25000, 40000, 50000, 20000,
        60000, 35000, 80000, 30000
    ],
    "loan_amount": [
        10000, 15000, 20000, 12000,
        25000, 10000, 30000, 18000
    ],
    "credit_score": [
        600, 700, 750, 550,
        800, 650, 780, 580
    ],
    "approved": [
        1, 1, 1, 0, 1, 1, 1, 0
    ]
}

df = pd.DataFrame(data)

X = df[["income", "loan_amount", "credit_score"]]
y = df["approved"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

model.fit(X, y)

# ==================================================
# 4. HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <p>FINANCIAL TECHNOLOGY • MACHINE LEARNING</p>
    <h1>💳 CrediSense</h1>
    <h3 style="color:#d8f4f5;">
        Intelligent Credit Scoring System
    </h3>
    <p>
        Explore an educational machine learning prototype
        for credit approval prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# 5. DASHBOARD METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MODEL TYPE", "Logistic Regression")

with col2:
    st.metric("INPUT FEATURES", "03")

with col3:
    st.metric("SAMPLE RECORDS", "08")

st.write("")

# ==================================================
# 6. APPLICANT INPUT
# ==================================================

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("👤 Applicant Details")
    st.caption("Enter financial details below.")

    with st.form("credit_form"):

        income = st.number_input(
            "Annual Income (₹)",
            min_value=10000,
            max_value=10000000,
            value=50000,
            step=5000
        )

        loan = st.number_input(
            "Requested Loan Amount (₹)",
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

        submitted = st.form_submit_button(
            "⚡ Analyse Credit Profile",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# 7. APPLICANT PROFILE
# ==================================================

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.subheader("📊 Credit Profile Overview")
    st.caption("Review your entered financial information.")

    st.markdown(
        f'<div class="small-label">Annual Income</div>'
        f'<div class="big-value">₹{income:,.0f}</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        f'<div class="small-label">Requested Loan</div>'
        f'<div class="big-value">₹{loan:,.0f}</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown(
        f'<div class="small-label">Credit Score</div>'
        f'<div class="big-value">{score} / 900</div>',
        unsafe_allow_html=True
    )

    st.progress((score - 300) / 600)

    st.caption("Credit score scale: 300–900")

    st.write("")

    ratio = (loan / income) * 100

    st.markdown(
        f'<div class="small-label">Loan-to-Income Ratio</div>'
        f'<div class="big-value">{ratio:.1f}%</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# ==================================================
# 8. MACHINE LEARNING PREDICTION
# ==================================================

if submitted:

    applicant = pd.DataFrame({
        "income": [income],
        "loan_amount": [loan],
        "credit_score": [score]
    })

    prediction = model.predict(applicant)[0]

    probability = model.predict_proba(applicant)[0][1]

    st.divider()

    st.subheader("🔍 Credit Analysis Result")

    result_col, probability_col = st.columns(
        [1.2, 1],
        gap="large"
    )

    with result_col:

        if prediction == 1:

            st.markdown(f"""
            <div class="result-approved">
                <p class="small-label">MODEL OUTPUT</p>
                <h2 style="color:white;">✓ Demo Approved</h2>
                <p style="color:#d8f4f5;">
                    The model predicts a positive credit approval outcome.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-rejected">
                <p class="small-label">MODEL OUTPUT</p>
                <h2 style="color:white;">✕ Demo Not Approved</h2>
                <p style="color:#f6dce1;">
                    The model predicts a negative credit approval outcome.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with probability_col:

        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.subheader("Approval Probability")

        st.markdown(
            f'<div class="big-value">{probability * 100:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.progress(float(probability))

        st.caption(
            "Model-estimated probability of the positive class."
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.subheader("📋 Applicant Summary")

    summary1, summary2, summary3 = st.columns(3)

    with summary1:
        st.metric("Annual Income", f"₹{income:,.0f}")

    with summary2:
        st.metric("Loan Amount", f"₹{loan:,.0f}")

    with summary3:
        st.metric("Credit Score", f"{score}/900")

# ==================================================
# 9. PROJECT INFORMATION
# ==================================================

st.divider()

st.subheader("💡 About CrediSense")

info1, info2 = st.columns(2)

with info1:
    st.markdown("""
    <div class="panel">
        <h4 style="color:white;">Technology Stack</h4>
        <p>🐍 Python</p>
        <p>📊 Pandas</p>
        <p>🤖 Scikit-learn</p>
        <p>🌐 Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="panel">
        <h4 style="color:white;">Project Features</h4>
        <p>✓ Applicant information input</p>
        <p>✓ Machine learning prediction</p>
        <p>✓ Probability visualization</p>
        <p>✓ Credit profile summary</p>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# 10. DISCLAIMER AND FOOTER
# ==================================================

st.warning(
    "Educational prototype only. This model uses just 8 sample "
    "records. Its predictions and probabilities are not validated "
    "for real-world lending or financial decisions."
)

st.markdown("""
<div class="footer">
    <b>CrediSense</b> | Credit Intelligence Project<br>
    Developed by Ashwini Shelke<br>
    Built with Python, Machine Learning and Streamlit
</div>
""", unsafe_allow_html=True)
