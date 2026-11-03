from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MediPredict | Insurance Estimator",
    page_icon="🏥",
    layout="wide"
)


# ==========================================
# MODEL LOADING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* ==========================================
   PAGE BACKGROUND & CONTAINER
========================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #D4F1EA,
        #A8DDD2,
        #E0F5EF
    );
}

.block-container {
    padding-top: 3.5rem;
    padding-bottom: 2.5rem;
    max-width: 1240px;
}


/* ==========================================
   HEADER (MEDIPREDICT CARD)
========================================== */

.header-box {
    background: linear-gradient(
        135deg,
        #FFE1EA,
        #F8CFE0,
        #FDEAF0
    );

    border: 1px solid #F5C5D6;
    border-radius: 20px;

    padding: 32px 24px;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 35px;

    box-shadow: 0 5px 18px rgba(135, 68, 95, 0.08);

    animation: fadeIn 0.7s ease;
    transition: all 0.3s ease;
}

/* Header hover */

.header-box:hover {
    transform: translateY(-4px);

    box-shadow:
        0 10px 25px rgba(135, 68, 95, 0.16);
}

.header-title {
    color: #87445F;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 8px;
}

.header-subtitle {
    color: #875D70;
    font-size: 15px;
    margin-bottom: 0;
}


/* ==========================================
   SECTION HEADINGS
========================================== */

.section-title {
    color: #145B50;
    font-size: 22px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 18px;
}


/* ==========================================
   FORM WRAPPER
========================================== */

[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.55);
    border: 1px solid rgba(155, 220, 197, 0.6);
    border-radius: 20px;
    padding: 24px 20px;
    box-shadow: 0 6px 20px rgba(20, 91, 80, 0.07);
}


/* ==========================================
   INPUT LABELS
========================================== */

.stSelectbox label,
.stNumberInput label {
    color: #205E53 !important;
    font-weight: 600 !important;
}


/* ==========================================
   NUMBER INPUT FIELD (UNIFIED WHITE CONTAINER)
========================================== */

div[data-testid="stNumberInputContainer"],
.stNumberInput div:has(> input) {
    background-color: #FFFFFF !important;
    border: 1px solid #9BDCC5 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    transition: all 0.25s ease !important;
}

div[data-testid="stNumberInputContainer"]:hover,
.stNumberInput div:has(> input):hover {
    border-color: #168B75 !important;

    box-shadow:
        0 0 0 3px rgba(22, 139, 117, 0.15) !important;
}

div[data-testid="stNumberInputContainer"]:focus-within,
.stNumberInput div:has(> input):focus-within {
    border-color: #168B75 !important;

    box-shadow:
        0 0 0 3px rgba(22, 139, 117, 0.20) !important;
}

.stNumberInput input {
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #205E53 !important;
}

div[data-testid="stNumberInputContainer"] > div {
    background-color: transparent !important;
    background: transparent !important;
}


/* ==========================================
   PLUS AND MINUS BUTTONS (NUMBER INPUT ONLY)
========================================== */

.stNumberInput button[data-testid^="stNumberInputStep"],
.stNumberInput button[aria-label="Increment"],
.stNumberInput button[aria-label="Decrement"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    transition: transform 0.2s ease, background-color 0.2s ease !important;
}

.stNumberInput button[data-testid^="stNumberInputStep"] svg,
.stNumberInput button[aria-label="Increment"] svg,
.stNumberInput button[aria-label="Decrement"] svg {
    color: #168B75 !important;
    fill: #168B75 !important;
    transform-origin: center;
    transition: transform 0.2s ease, color 0.2s ease, fill 0.2s ease !important;
}

.stNumberInput button[data-testid^="stNumberInputStep"]:hover,
.stNumberInput button[aria-label="Increment"]:hover,
.stNumberInput button[aria-label="Decrement"]:hover {
    background: transparent !important;
    background-color: transparent !important;
    transform: scale(1.08);
}

.stNumberInput button[data-testid^="stNumberInputStep"]:hover svg,
.stNumberInput button[aria-label="Increment"]:hover svg,
.stNumberInput button[aria-label="Decrement"]:hover svg {
    color: #08745F !important;
    fill: #08745F !important;
    transform: scale(1.15);
}

.stNumberInput button[data-testid^="stNumberInputStep"]:active,
.stNumberInput button[aria-label="Increment"]:active,
.stNumberInput button[aria-label="Decrement"]:active {
    transform: scale(0.92) !important;
}


/* ==========================================
   DROPDOWN FIELD
========================================== */

.stSelectbox [data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div:has(> input) {
    background-color: #FFFFFF !important;
    border: 1px solid #9BDCC5 !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
}

.stSelectbox input {
    color: #205E53 !important;
}

.stSelectbox [data-baseweb="select"]:hover > div,
div[data-testid="stSelectbox"] div:has(> input):hover {
    border-color: #168B75 !important;

    box-shadow:
        0 0 0 3px rgba(22, 139, 117, 0.15) !important;
}

.stSelectbox [data-baseweb="select"]:focus-within > div,
div[data-testid="stSelectbox"] div:has(> input)[data-focus-within],
div[data-testid="stSelectbox"] div:has(> input):focus-within {
    border-color: #168B75 !important;

    box-shadow:
        0 0 0 3px rgba(22, 139, 117, 0.20) !important;
}


/* ==========================================
   DROPDOWN ARROW & RIPPLE
========================================== */

@keyframes buttonRipple {
    0% {
        box-shadow: 0 0 0 0 rgba(22, 139, 117, 0.35),
                    0 0 0 6px rgba(22, 139, 117, 0.20),
                    0 0 0 12px rgba(22, 139, 117, 0.08);
    }
    100% {
        box-shadow: 0 0 0 6px rgba(22, 139, 117, 0.20),
                    0 0 0 12px rgba(22, 139, 117, 0.08),
                    0 0 0 18px rgba(22, 139, 117, 0);
    }
}

.stSelectbox button[aria-label="Open"] {
    background: transparent !important;
    border: none !important;
    border-radius: 50% !important;
    box-shadow: none !important;
    cursor: pointer !important;
    transition: background-color 0.25s ease, box-shadow 0.25s ease !important;
}

.stSelectbox button[aria-label="Open"]:hover {
    background: rgba(22, 139, 117, 0.08) !important;
}

.stSelectbox button[aria-label="Open"][aria-expanded="true"] {
    background: rgba(22, 139, 117, 0.12) !important;
    animation: buttonRipple 0.8s ease-out infinite !important;
}

.stSelectbox button[aria-label="Open"] svg {
    color: #168B75 !important;
    fill: #168B75 !important;

    transform: rotate(0deg) scale(1);
    transform-origin: center;

    transition:
        transform 0.3s ease,
        color 0.25s ease,
        fill 0.25s ease !important;
}

.stSelectbox button[aria-label="Open"]:hover svg {
    color: #08745F !important;
    fill: #08745F !important;
    transform: scale(1.15);
}

.stSelectbox button[aria-label="Open"][aria-expanded="true"] svg {
    color: #08745F !important;
    fill: #08745F !important;
    transform: rotate(180deg);
}

.stSelectbox button[aria-label="Open"]:active svg {
    transform: scale(0.9);
}


/* ==========================================
   DROPDOWN MENU ANIMATION & OPTIONS
========================================== */

@keyframes dropdownFadeIn {
    from {
        opacity: 0;
    }

    to {
        opacity: 1;
    }
}

@keyframes dropdownSlideDown {
    from {
        opacity: 0;
        transform: translateY(-16px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Floating overlay container */
[data-testid="stSelectboxVirtualDropdown"] {
    animation: dropdownFadeIn 0.3s ease both;
    background: #FFFFFF !important;
    border: 1px solid #9BDCC5 !important;
    border-radius: 12px !important;
    box-shadow:
        0 4px 16px rgba(18, 107, 85, 0.12),
        0 1px 4px rgba(0, 0, 0, 0.05) !important;
    overflow: hidden !important;
}

/* Inner listbox container slides down smoothly (0.3s) */
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
[role="listbox"] {
    animation: dropdownSlideDown 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
    transform-origin: top center;
    background: #FFFFFF !important;
}

/* Individual options with clean dividers and smooth transitions */
[role="option"] {
    background: #FFFFFF !important;
    color: #205E53 !important;
    border-bottom: 1px solid rgba(20, 91, 80, 0.07) !important;
    transition: background-color 0.25s ease, color 0.25s ease !important;
    font-size: 14px !important;
}

[role="option"]:last-child {
    border-bottom: none !important;
}

[role="option"] > div,
[role="option"] [data-item-hl] {
    transition:
        background-color 0.25s ease,
        color 0.25s ease !important;
}

/* Option Hover State */
[role="option"]:hover,
[role="option"][data-hovered],
[role="option"][data-focused] {
    background-color: #E8FAF2 !important;
    color: #168B75 !important;
}

[role="option"]:hover [data-item-hl],
[role="option"][data-hovered] [data-item-hl],
[role="option"][data-focused] [data-item-hl] {
    background-color: #E8FAF2 !important;
    color: #168B75 !important;
}

/* Option Active / Selected State */
[role="option"][aria-selected="true"] {
    background-color: #D9F6E9 !important;
    color: #126B55 !important;
    font-weight: 700 !important;
}

[role="option"][aria-selected="true"] [data-item-hl] {
    background-color: #D9F6E9 !important;
    color: #126B55 !important;
}


/* ==========================================
   SUBMIT BUTTON
========================================== */

.stFormSubmitButton button {
    background: linear-gradient(
        135deg,
        #168B75,
        #0B705F
    ) !important;

    color: white !important;

    border: none !important;
    border-radius: 12px !important;

    font-weight: 700 !important;
    padding: 14px 20px !important;
    font-size: 16px !important;

    transition: all 0.3s ease !important;
}

.stFormSubmitButton button:hover {
    background: linear-gradient(
        135deg,
        #0B705F,
        #08745F
    ) !important;

    transform: translateY(-3px);

    box-shadow:
        0 6px 15px rgba(11, 112, 95, 0.25);
}

.stFormSubmitButton button:active {
    transform: scale(0.97);
}


/* ==========================================
   PREDICTION RESULT BOX (BIGGER CARD)
========================================== */

.result-box {
    background: linear-gradient(
        135deg,
        #D9F6E9,
        #BCEBDD,
        #E8FAF2
    );

    border: 2px solid #9BDCC5;
    border-radius: 24px;

    padding: 45px 30px;
    text-align: center;
    min-height: 380px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    box-shadow:
        0 10px 28px rgba(18, 107, 85, 0.14);

    animation: resultAppear 0.6s ease;
    transition: all 0.3s ease;
}

/* Result card hover */

.result-box:hover {
    transform: translateY(-4px);

    box-shadow:
        0 14px 32px rgba(18, 107, 85, 0.18);
}

.result-placeholder-card {
    background: rgba(255, 255, 255, 0.45);
    border: 2px dashed #9BDCC5;
    border-radius: 24px;
    padding: 45px 30px;
    text-align: center;
    min-height: 380px;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    box-shadow:
        0 4px 16px rgba(18, 107, 85, 0.05);
}

.result-label {
    color: #347566;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
}

.result-value {
    color: #126B55;
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -0.5px;
}


/* ==========================================
   FOOTER
========================================== */

.footer {
    color: #47766D;

    text-align: center;
    font-size: 13px;

    margin-top: 45px;
    padding: 20px;
}


/* ==========================================
   ANIMATIONS
========================================== */

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(12px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes resultAppear {
    from {
        opacity: 0;
        transform: translateY(15px) scale(0.96);
    }

    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes resultAppearAlt {
    from {
        opacity: 0;
        transform: translateY(15px) scale(0.96);
    }

    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}


/* ==========================================
   MOBILE RESPONSIVENESS
========================================== */

@media (max-width: 800px) {

    .header-title {
        font-size: 30px;
    }

    .header-subtitle {
        font-size: 13px;
    }

    .result-value {
        font-size: 36px;
    }
    
    .result-box, .result-placeholder-card {
        min-height: auto;
        padding: 30px 20px;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER (WITH TOP PADDING)
# ==========================================

st.markdown(
    """<div class="header-box">
    <h1 class="header-title" style="color: #87445F; font-size: 38px; font-weight: 800; margin: 0 0 8px 0;">MediPredict 🏥</h1>
    <p class="header-subtitle" style="color: #875D70; font-size: 15px; margin: 0;">Estimate your medical insurance costs using Machine Learning</p>
    </div>""",
    unsafe_allow_html=True
)


# ==========================================
# 2-COLUMN SPLIT: LEFT HALF (INPUTS) & RIGHT HALF (PREDICTION)
# ==========================================

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:

    st.markdown(
        '<div class="section-title">Enter Your Details</div>',
        unsafe_allow_html=True
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25,
                step=1
            )

            sex = st.selectbox(
                "Sex",
                ["male", "female"]
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1
            )

        with col2:

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )

            smoker = st.selectbox(
                "Smoker",
                ["no", "yes"]
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

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "Calculate Insurance Estimate",
            use_container_width=True
        )


with right_col:

    st.markdown(
        '<div class="section-title">Insurance Prediction</div>',
        unsafe_allow_html=True
    )

    if "prediction_count" not in st.session_state:
        st.session_state.prediction_count = 0

    if submitted:
        st.session_state.prediction_count += 1
        card_id = f"prediction-result-{st.session_state.prediction_count}"
        anim_name = "resultAppear" if st.session_state.prediction_count % 2 == 1 else "resultAppearAlt"

        input_data = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region]
            }
        )

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""<div id="{card_id}" class="result-box {card_id}" style="animation: {anim_name} 0.6s ease;">
            <div style="font-size: 38px; margin-bottom: 8px;">📋</div>
            <p class="result-label" style="color: #347566; font-size: 18px; font-weight: 600; margin: 0 0 10px 0;">Estimated Medical Insurance Charges</p>
            <div class="result-value" style="color: #126B55; font-size: 52px; font-weight: 800; margin: 0 0 15px 0;">${prediction:,.2f}</div>
            <div style="color: #205E53; font-size: 14px; font-weight: 500; background: rgba(22, 139, 117, 0.12); padding: 8px 18px; border-radius: 20px; display: inline-block;">
            {age} yrs • {sex} • BMI: {bmi:.1f} • Smoker: {smoker}
            </div>
            </div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<div class="result-placeholder-card">
            <div style="font-size: 44px; margin-bottom: 12px;">🩺</div>
            <h3 style="color: #145B50; font-size: 22px; font-weight: 700; margin: 0 0 10px 0;">Ready for Estimation</h3>
            <p style="color: #347566; font-size: 15px; margin: 0; line-height: 1.5; max-width: 340px;">
            Enter your personal details on the left and click <strong>Calculate Insurance Estimate</strong> to view your predicted charges.
            </p>
            </div>""",
            unsafe_allow_html=True
        )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """<div class="footer">
    MediPredict | Machine Learning Project<br><br>
    This prediction is for educational purposes only and should not be considered professional financial advice.
    </div>""",
    unsafe_allow_html=True
)
