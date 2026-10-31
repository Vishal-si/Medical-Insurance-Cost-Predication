
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="APEX LIFE INSURANCE",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
}

.main-title {
    font-size: 35px;
    font-weight: bold;
    color: #15345b;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
}

div.stButton > button {
    background-color: #1769aa;
    color: white;
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL AND SCALER ----------------

@st.cache_resource
def load_model():

    currentPath = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(currentPath, "model.pkl")
    scaler_path = os.path.join(currentPath, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("model.pkl file not found!")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler.pkl file not found!")

    def load_file(path):
        try:
            return pickle.load(open(path, "rb"))
        except (pickle.UnpicklingError, EOFError, ValueError):
            return joblib.load(path)

    model = load_file(model_path)
    scaler = load_file(scaler_path)

    return model, scaler


try:
    model, scaler = load_model()
    st.sidebar.success("Model and Scaler Loaded Successfully")

except Exception as e:
    st.error(f"Unable to load model or scaler: {e}")
    st.info(
        "Check whether your files were saved using Pickle or Joblib "
        "and verify that both files are valid."
    )
    st.stop()


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🏥 APEX LIFE INSURANCE</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Insurance Charges Estimation</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- SIDEBAR ----------------

st.sidebar.title("🏥 Insurance System")

st.sidebar.write(
    "Welcome to the APEX LIFE INSURANCE System."
)

st.sidebar.info(
    "Enter customer information to estimate medical insurance charges."
)


# ---------------- CUSTOMER INPUT ----------------

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

with col2:

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    children = st.selectbox(
        "Number of Children",
        list(range(11))
    )

with col3:

    smoker = st.selectbox(
        "Smoking Status",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "southwest",
            "southeast",
            "northwest",
            "northeast"
        ]
    )


st.divider()


# ---------------- CUSTOMER DETAILS ----------------

st.header("📋 Customer Details")

details = pd.DataFrame({
    "Feature": [
        "Age",
        "Gender",
        "BMI",
        "Children",
        "Smoker",
        "Region"
    ],

    "Entered Value": [
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    ]
})

st.dataframe(
    details,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ---------------- PREDICTION ----------------

st.header("💰 Insurance Charges Prediction")

if st.button("🔍 Predict Insurance Charges"):

    try:

        # Create input dataframe
        input_data = pd.DataFrame({

            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]

        })

        # One hot encoding
        input_data = pd.get_dummies(input_data)

        # Match training feature columns
        if hasattr(scaler, "feature_names_in_"):

            input_data = input_data.reindex(
                columns=scaler.feature_names_in_,
                fill_value=0
            )

        # Scale input data
        scaled_data = scaler.transform(input_data)

        # Predict insurance charges
        prediction = model.predict(scaled_data)[0]

        # Display result
        st.success("Prediction Completed Successfully!")

        st.subheader("Estimated Medical Insurance Charges")

        st.metric(
            label="Predicted Insurance Charges",
            value=f"{prediction:,.2f}"
        )

        st.balloons()

    except Exception as e:

        st.error(f"Prediction Error: {e}")

        st.warning(
            "Please check whether the input preprocessing "
            "matches your model training preprocessing."
        )


# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    """
    <center>
        <p style="color:gray;">
            APEX LIFE INSURANCE Prediction System |
            Powered by Machine Learning
        </p>
    </center>
    """,
    unsafe_allow_html=True
)
