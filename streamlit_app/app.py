import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Risk Models", page_icon="📊", layout="centered")

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

SEGMENTS = ["retail", "small_business", "corporate", "private_banking"]  # risk-ordered, same as notebook
INDUSTRIES = [  # risk-ordered, same as notebook
    "construction", "software", "import export", "education",
    "professional services", "healthcare", "real estate",
    "hospitality", "agriculture", "retail trade", "logistics", "manufacturing",
]
RF_FEATURES = [  # exact column order the RF was trained on
    "annual_income", "expected_monthly_volume", "volume_to_income",
    "log_income", "log_volume", "log_ratio",
    "monthly_income_share", "log_monthly_income_share",
    "segment_ord", "industry_ord", "country_risk_score",
]


@st.cache_resource
def load_pickle(path):
    return joblib.load(path)


def show_prediction(classes, proba):
    classes = list(classes)
    names = [RISK_LABELS.get(int(c), str(c)) if str(c).lstrip("-").isdigit() else str(c) for c in classes]
    best = int(np.argmax(proba))
    label = names[best]
    st.success(f"{RISK_COLORS.get(label, '')} Prediction: **{label}**  ({proba[best]:.1%})")
    st.bar_chart(pd.DataFrame({"probability": proba}, index=names))


# ---------------------------------------------------------------- RF page
def page_random_forest():
    st.header("Random Forest — Customer Risk")
    try:
        art = load_pickle("rf_risk_model.pkl")
    except FileNotFoundError:
        st.error("rf_risk_model.pkl not found. Run the save cell at the end of your notebook first.")
        return
    model, country_risk = art["model"], art["country_risk"]

    c1, c2, c3 = st.columns(3)
    segment = c1.selectbox("Segment", SEGMENTS)
    industry = c2.selectbox("Industry", INDUSTRIES)
    country = c3.selectbox("Country", sorted(country_risk))
    annual_income = st.number_input("Annual income", min_value=1.0, value=50000.0, step=1000.0)
    monthly_volume = st.number_input("Expected monthly volume", min_value=1.0, value=2500.0, step=100.0)

    if st.button("Predict", key="rf_btn"):
        vti = monthly_volume * 12 / annual_income  # matches dataset's volume_to_income
        share = monthly_volume / (annual_income + 1)
        row = {
            "annual_income": annual_income,
            "expected_monthly_volume": monthly_volume,
            "volume_to_income": vti,
            "log_income": np.log1p(annual_income),
            "log_volume": np.log1p(monthly_volume),
            "log_ratio": np.log1p(vti),
            "monthly_income_share": share,
            "log_monthly_income_share": np.log1p(share),
            "segment_ord": SEGMENTS.index(segment),
            "industry_ord": INDUSTRIES.index(industry),
            "country_risk_score": country_risk[country],
        }
        X = pd.DataFrame([row])[RF_FEATURES]
        show_prediction(model.classes_, model.predict_proba(X)[0])


# ---------------------------------------------------------------- XGBoost page (fraud)
TX_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]  # CASH_IN = dropped dummy (drop_first=True)


@st.cache_resource
def load_fraud_artifacts():
    base = "model_files"
    return (
        joblib.load(f"{base}/xgboost_fraud_model.pkl"),
        joblib.load(f"{base}/robust_scaler.pkl"),
        joblib.load(f"{base}/model_columns.pkl"),
    )


def page_xgboost():
    st.header("XGBoost — Transaction Fraud Detection")
    try:
        model, scaler, columns = load_fraud_artifacts()
    except FileNotFoundError:
        st.error("model_files/ not found. Put the model_files folder (xgboost_fraud_model.pkl, "
                 "robust_scaler.pkl, model_columns.pkl) next to app.py.")
        return

    defaults = {"fx_type": "PAYMENT", "fx_step": 1, "fx_amount": 1000.0, "fx_old_org": 5000.0,
                "fx_new_org": 4000.0, "fx_old_dest": 0.0, "fx_new_dest": 1000.0}
    examples = {
        "Normal payment": defaults,
        "Fraud example (TRANSFER, drains account)": {"fx_type": "TRANSFER", "fx_step": 1, "fx_amount": 181.0,
            "fx_old_org": 181.0, "fx_new_org": 0.0, "fx_old_dest": 0.0, "fx_new_dest": 0.0},
        "Fraud example (CASH_OUT, drains account)": {"fx_type": "CASH_OUT", "fx_step": 1, "fx_amount": 181.0,
            "fx_old_org": 181.0, "fx_new_org": 0.0, "fx_old_dest": 21182.0, "fx_new_dest": 0.0},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    def load_example():
        st.session_state.update(examples[st.session_state["fx_choice"]])

    st.selectbox("Load an example", list(examples), key="fx_choice", on_change=load_example)

    c1, c2 = st.columns(2)
    tx_type = c1.selectbox("Transaction type", TX_TYPES, key="fx_type")
    step = c2.number_input("Step (hour)", min_value=1, step=1, key="fx_step")
    amount = st.number_input("Amount", min_value=0.0, step=100.0, key="fx_amount")

    c3, c4 = st.columns(2)
    old_org = c3.number_input("Sender balance before", min_value=0.0, step=100.0, key="fx_old_org")
    new_org = c4.number_input("Sender balance after", min_value=0.0, step=100.0, key="fx_new_org")
    c5, c6 = st.columns(2)
    old_dest = c5.number_input("Receiver balance before", min_value=0.0, step=100.0, key="fx_old_dest")
    new_dest = c6.number_input("Receiver balance after", min_value=0.0, step=100.0, key="fx_new_dest")

    if st.button("Check transaction", key="xgb_btn"):
        row = {
            "step": step,
            "amount": amount,
            "oldbalanceOrg": old_org,
            "newbalanceOrig": new_org,
            "oldbalanceDest": old_dest,
            "newbalanceDest": new_dest,
            # same engineered features as the notebook
            "errorBalanceOrg": new_org + amount - old_org,
            "errorBalanceDest": old_dest + amount - new_dest,
        }
        for t in TX_TYPES[1:]:
            row[f"type_{t}"] = 1.0 if tx_type == t else 0.0

        X = pd.DataFrame([row]).reindex(columns=list(columns), fill_value=0.0).astype(float)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        is_fraud = int(model.predict(X_scaled)[0]) == 1

        if is_fraud:
            st.error("🔴(Fraud)")
        else:
            st.success("🟢(Normal)")


# ---------------------------------------------------------------- navigation
st.sidebar.title("Models")
choice = st.sidebar.radio("Choose a model", ["Random Forest", "XGBoost"])
page_random_forest() if choice == "Random Forest" else page_xgboost()
