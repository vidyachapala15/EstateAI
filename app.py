import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="EstateAI | Property Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* Main container */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        padding: 2.5rem;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #111827,
            #312e81
        );
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    }

    .hero-title {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: #c7d2fe;
        font-size: 1.15rem;
        margin-bottom: 0;
    }

    /* Section headings */
    .section-title {
        font-size: 1.5rem;
        font-weight: 750;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 25px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(
            135deg,
            #312e81,
            #4338ca
        );
        padding: 2.2rem;
        border-radius: 22px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 40px rgba(49,46,129,0.25);
        margin-top: 1rem;
    }

    .result-label {
        font-size: 1rem;
        color: #c7d2fe;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-price {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0.5rem 0;
    }

    .result-note {
        color: #e0e7ff;
        font-size: 0.95rem;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 1.3rem;
        border: 1px solid #e5e7eb;
        text-align: center;
        height: 100%;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
    }

    .feature-icon {
        font-size: 2rem;
    }

    .feature-title {
        font-weight: 700;
        color: #111827;
        margin-top: 0.5rem;
    }

    .feature-text {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem;
        font-size: 1.05rem;
        font-weight: 700;
        border: none;
        background: #4338ca;
        color: white;
    }

    .stButton > button:hover {
        background: #312e81;
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e5e7eb;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "model/house_price_model.pkl"
    )


model = load_model()



# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🏠 EstateAI</div>
        <div class="hero-subtitle">
            AI-Powered Real Estate Valuation<br>
            Estimate property value using Machine Learning
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# ==========================================================
# INTRODUCTION
# ==========================================================

st.markdown("""
<div class="card">

<h2>Smart Property Valuation</h2>

<p>
EstateAI uses a Machine Learning model trained on historical
property data to estimate the potential selling price of a house.
Enter the property characteristics below and generate an
instant AI-powered valuation.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# PROPERTY INPUT SECTION
# ==========================================================

st.markdown(
    '<div class="section-title">🏡 Property Details</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=7,
        help="Overall material and finish quality of the house."
    )

    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=1800,
        step=50
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=10,
        value=3,
        step=1
    )


with col2:

    bathrooms = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=6,
        value=2,
        step=1
    )

    garage = st.number_input(
        "Garage Capacity",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    basement = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50
    )


with col3:

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2010,
        step=1
    )

    st.metric(
        "Property Age",
        f"{2026 - year_built} years"
    )

    st.metric(
        "Living Area",
        f"{living_area:,} sq ft"
    )


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "✨ Estimate Property Value"
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [living_area],
        "GarageCars": [garage],
        "TotalBsmtSF": [basement],
        "FullBath": [bathrooms],
        "BedroomAbvGr": [bedrooms],
        "YearBuilt": [year_built]
    })


    prediction = model.predict(
        input_data
    )[0]


    # ------------------------------------------------------
    # RESULT
    # ------------------------------------------------------

    st.markdown(f"""
    <div class="result-card">

        <div class="result-label">
            Estimated Property Value
        </div>

        <div class="result-price">
            ${prediction:,.0f}
        </div>

        <div class="result-note">
            AI-generated estimate based on the provided
            property characteristics.
        </div>

    </div>
    """, unsafe_allow_html=True)


    # ======================================================
    # ANALYTICS
    # ======================================================

    st.markdown(
        '<div class="section-title">📊 Property Analytics</div>',
        unsafe_allow_html=True
    )


    chart_data = pd.DataFrame({
        "Feature": [
            "Overall Quality",
            "Living Area",
            "Bedrooms",
            "Bathrooms",
            "Garage",
            "Basement"
        ],
        "Value": [
            overall_qual,
            living_area / 500,
            bedrooms,
            bathrooms,
            garage,
            basement / 500
        ]
    })


    fig = go.Figure(
        go.Bar(
            x=chart_data["Feature"],
            y=chart_data["Value"],
            text=chart_data["Value"].round(1),
            textposition="auto"
        )
    )


    fig.update_layout(
        title="Property Feature Overview",
        xaxis_title="Feature",
        yaxis_title="Normalized Value",
        template="plotly_white",
        height=420
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ======================================================
    # PROPERTY SUMMARY
    # ======================================================

    st.markdown(
        '<div class="section-title">🧠 AI Property Insights</div>',
        unsafe_allow_html=True
    )


    insight1, insight2, insight3 = st.columns(3)


    with insight1:

        st.markdown(f"""
        <div class="feature-card">

            <div class="feature-icon">🏠</div>

            <div class="feature-title">
                Property Quality
            </div>

            <div class="feature-text">
                Quality rating: {overall_qual}/10
            </div>

        </div>
        """, unsafe_allow_html=True)


    with insight2:

        st.markdown(f"""
        <div class="feature-card">

            <div class="feature-icon">📐</div>

            <div class="feature-title">
                Living Space
            </div>

            <div class="feature-text">
                {living_area:,} sq ft
            </div>

        </div>
        """, unsafe_allow_html=True)


    with insight3:

        st.markdown(f"""
        <div class="feature-card">

            <div class="feature-icon">🚗</div>

            <div class="feature-title">
                Garage
            </div>

            <div class="feature-text">
                Capacity: {garage} cars
            </div>

        </div>
        """, unsafe_allow_html=True)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div class="card">

<h3>🤖 About the AI Model</h3>

<p>
EstateAI uses a Random Forest Regression model trained on
historical residential property data. The prototype uses
property quality, living area, garage capacity, basement area,
bathrooms, bedrooms, and construction year as prediction
features.
</p>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">

<strong>EstateAI</strong> · AI-Powered Real Estate Valuation

<br><br>

Built as an IBM Project Prototype using
Python, Streamlit and Machine Learning.

</div>
""", unsafe_allow_html=True)