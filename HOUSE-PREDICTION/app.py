
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

# Get the directory where app.py is located
BASE_DIR = Path(__file__).resolve().parent

# Construct the absolute path to the model
MODEL_PATH = BASE_DIR / "best_housing_price_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()

    return joblib.load(MODEL_PATH)

model = load_model()

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        color: #2563eb;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .prediction-box {
        background-color: #ecfdf5;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #10b981;
    }

    .prediction-label {
        font-size: 20px;
        color: #065f46;
    }

    .prediction-value {
        font-size: 36px;
        font-weight: bold;
        color: #047857;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🏠 House Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict house prices using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("About the Model")

st.sidebar.info(
    """
    **Algorithm:** Multiple Linear Regression

    **Library:** Scikit-learn

    **Dataset:** Housing Price Dataset

    **Features:** 5

    **Target:** House Price
    """
)

st.sidebar.markdown("---")

st.sidebar.write("### Model Performance")

st.sidebar.metric(
    label="R² Score",
    value="0.5464"
)

st.sidebar.caption(
    "Performance measured on the held-out test dataset."
)

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("Enter Property Details")

st.write(
    "Provide the following information to estimate "
    "the selling price of your house."
)

col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area (sq. ft.)",
        min_value=100,
        max_value=50000,
        value=5000,
        step=100
    )

    bedrooms = st.number_input(
        "Number of Bedrooms",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    bathrooms = st.number_input(
        "Number of Bathrooms",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

with col2:

    stories = st.number_input(
        "Number of Stories",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    parking = st.number_input(
        "Parking Spaces",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

st.divider()

# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "Predict House Price",
    type="primary",
    use_container_width=True
):

    # Prepare input in the exact training feature order
    input_data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking
    }])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success("Prediction Generated Successfully!")

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="prediction-label">
                Estimated House Price
            </div>
            <div class="prediction-value">
                ₹{prediction:,.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Display input summary
    st.subheader("Prediction Summary")

    summary = pd.DataFrame({
        "Property Feature": [
            "Area",
            "Bedrooms",
            "Bathrooms",
            "Stories",
            "Parking"
        ],
        "Input Value": [
            f"{area} sq. ft.",
            bedrooms,
            bathrooms,
            stories,
            parking
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # Download prediction report
    report = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Stories": stories,
        "Parking": parking,
        "Predicted Price": round(prediction, 2)
    }])

    st.download_button(
        label="Download Prediction Report",
        data=report.to_csv(index=False),
        file_name="house_price_prediction.csv",
        mime="text/csv"
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    "<center>Developed using Python, Streamlit "
    "and Scikit-learn</center>",
    unsafe_allow_html=True
)