from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# PAGE CONFIGURATION
st.set_page_config(
    page_title="MediPredict | Insurance Estimator",
    page_icon="🏥",
    layout="wide"
)


# MODEL LOADING
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
   WHAT-IF SCENARIOS SIMULATOR & EXPANDER
========================================== */

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.60) !important;
    border: 1.5px solid rgba(155, 220, 197, 0.85) !important;
    border-radius: 20px !important;
    box-shadow: 0 6px 20px rgba(20, 91, 80, 0.07) !important;
    overflow: hidden !important;
    margin-top: 25px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stExpander"]:hover {
    box-shadow: 0 10px 24px rgba(20, 91, 80, 0.10) !important;
    border-color: #168B75 !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] summary:active {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #145B50 !important;
    padding: 18px 24px !important;
    border-radius: 18px 18px 0 0 !important;
    border: none !important;
    transition: background-color 0.25s ease !important;
}

[data-testid="stExpander"] summary:hover {
    background: #E8FAF2 !important;
    background-color: #E8FAF2 !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {
    color: #145B50 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

[data-testid="stExpander"] summary svg {
    color: #168B75 !important;
    fill: #168B75 !important;
}

[data-testid="stExpanderDetails"] {
    background: rgba(255, 255, 255, 0.60) !important;
    padding: 20px 24px 28px 24px !important;
    border-top: 1px solid rgba(155, 220, 197, 0.4) !important;
}

.unchanged-badge {
    background: rgba(22, 139, 117, 0.08);
    border: 1px solid #9BDCC5;
    border-left: 5px solid #168B75;
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 20px;
    color: #205E53;
    font-size: 14px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}

.scenario-intro {
    color: #347566;
    font-size: 15px;
    margin-bottom: 16px;
    line-height: 1.5;
}

.scenario-banner-increased {
    background: linear-gradient(135deg, #FFE1EA, #F8CFE0, #FDEAF0);
    border: 1.5px solid #F5C5D6;
    border-radius: 16px;
    padding: 16px 22px;
    margin-top: 24px;
    margin-bottom: 20px;
    color: #87445F;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 14px rgba(135, 68, 95, 0.08);
    animation: fadeIn 0.4s ease;
}

.scenario-banner-decreased {
    background: linear-gradient(135deg, #D9F6E9, #BCEBDD, #E8FAF2);
    border: 1.5px solid #9BDCC5;
    border-radius: 16px;
    padding: 16px 22px;
    margin-top: 24px;
    margin-bottom: 20px;
    color: #126B55;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 14px rgba(18, 107, 85, 0.08);
    animation: fadeIn 0.4s ease;
}

.scenario-banner-neutral {
    background: #FFFFFF;
    border: 1.5px solid #9BDCC5;
    border-radius: 16px;
    padding: 16px 22px;
    margin-top: 24px;
    margin-bottom: 20px;
    color: #205E53;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 14px rgba(20, 91, 80, 0.05);
    animation: fadeIn 0.4s ease;
}

.comparison-card {
    background: #FFFFFF;
    border: 1.5px solid #9BDCC5;
    border-radius: 16px;
    padding: 20px 14px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(20, 91, 80, 0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.comparison-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(20, 91, 80, 0.12);
    border-color: #168B75;
}

.comparison-label {
    color: #347566;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.comparison-value {
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    word-break: break-word;
}

.comparison-caption {
    color: #47766D;
    font-size: 12px;
    font-weight: 500;
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
    st.markdown('<div class="section-title">Enter Your Details</div>', unsafe_allow_html=True)

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

            sex = st.selectbox("Sex",["male", "female"])

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

            smoker = st.selectbox("Smoker",["no", "yes"])

            region = st.selectbox("Region",["southwest", "southeast", "northwest", "northeast"])

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Calculate Insurance Estimate", use_container_width=True)

    if submitted:
        st.session_state.prediction_count += 1
        prev_ver = st.session_state.get("scenario_version", 0)
        st.session_state.scenario_version = prev_ver + 1
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
        prediction = float(model.predict(input_data)[0])
        st.session_state.original_prediction = prediction
        st.session_state.original_input = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }
        # Reset scenario comparison when a new original estimate is calculated
        st.session_state.scenario_prediction = None
        st.session_state.scenario_input = None
        for key in [
            f"whatif_input_age_{prev_ver}",
            f"whatif_input_bmi_{prev_ver}",
            f"whatif_input_smoker_{prev_ver}",
            f"whatif_input_children_{prev_ver}",
            f"whatif_input_region_{prev_ver}",
            "whatif_input_age",
            "whatif_input_bmi",
            "whatif_input_smoker",
            "whatif_input_children",
            "whatif_input_region",
        ]:
            st.session_state.pop(key, None)


with right_col:
    st.markdown('<div class="section-title">Insurance Prediction</div>', unsafe_allow_html=True)

    if "prediction_count" not in st.session_state:
        st.session_state.prediction_count = 0

    if st.session_state.get("original_prediction") is not None:
        orig_pred = st.session_state.original_prediction
        orig_info = st.session_state.original_input
        card_id = f"prediction-result-{st.session_state.prediction_count}"
        anim_name = "resultAppear" if st.session_state.prediction_count % 2 == 1 else "resultAppearAlt"

        st.markdown(
            f"""<div id="{card_id}" class="result-box {card_id}" style="animation: {anim_name} 0.6s ease;">
            <div style="font-size: 38px; margin-bottom: 8px;">📋</div>
            <p class="result-label" style="color: #347566; font-size: 18px; font-weight: 600; margin: 0 0 10px 0;">Estimated Medical Insurance Charges</p>
            <div class="result-value" style="color: #126B55; font-size: 52px; font-weight: 800; margin: 0 0 15px 0;">${orig_pred:,.2f}</div>
            <div style="color: #205E53; font-size: 14px; font-weight: 500; background: rgba(22, 139, 117, 0.12); padding: 8px 18px; border-radius: 20px; display: inline-block;">
            {orig_info['age']} yrs • {orig_info['sex']} • BMI: {orig_info['bmi']:.1f} • Smoker: {orig_info['smoker']}
            </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            """<div class="result-placeholder-card">
            <div style="font-size: 44px; margin-bottom: 12px;">🩺</div>
            <h3 style="color: #145B50; font-size: 22px; font-weight: 700; margin: 0 0 10px 0;">Ready for Estimation</h3>
            <p style="color: #347566; font-size: 15px; margin: 0; line-height: 1.5; max-width: 340px;">
            Enter your personal details on the left and click <strong>Calculate Insurance Estimate</strong> to view your predicted charges.
            </p>
            </div>""", unsafe_allow_html=True)


# ==========================================
# EXPLORE WHAT-IF SCENARIOS SIMULATOR
# ==========================================

if st.session_state.get("original_prediction") is not None:
    orig = st.session_state.original_input
    orig_pred = st.session_state.original_prediction

    st.markdown('<div style="margin-top: 35px;"></div>', unsafe_allow_html=True)

    with st.expander("Explore What-if Scenarios", expanded=True):
        st.markdown(
            f"""<div class="scenario-intro">
            Simulate how modifying personal factors (Age, BMI, Smoking status, Number of Children, Region) affects your estimated medical insurance cost while keeping your biological sex constant.
            </div>
            <div class="unchanged-badge">
            <span style="font-weight: 700; color: #145B50;">🔒 Unchanged Profile Detail:</span>
            &nbsp;&nbsp;
            <span>Sex: <strong>{orig['sex'].title()}</strong></span>
            </div>""", unsafe_allow_html=True)

        scenario_ver = st.session_state.get("scenario_version", 0)

        with st.form("whatif_form"):
            w_col1, w_col2, w_col3 = st.columns(3)

            with w_col1:
                whatif_age = st.number_input(
                    "What-if Age",
                    min_value=int(orig["age"]),
                    max_value=100,
                    value=int(orig["age"]),
                    step=1,
                    key=f"whatif_input_age_{scenario_ver}"
                )

            with w_col2:
                whatif_bmi = st.number_input(
                    "What-if BMI",
                    min_value=10.0,
                    max_value=60.0,
                    value=float(orig["bmi"]),
                    step=0.1,
                    key=f"whatif_input_bmi_{scenario_ver}"
                )

            with w_col3:
                whatif_smoker = st.selectbox(
                    "What-if Smoker",
                    ["no", "yes"],
                    index=0 if orig["smoker"] == "no" else 1,
                    key=f"whatif_input_smoker_{scenario_ver}"
                )

            w_col4, w_col5 = st.columns(2)

            with w_col4:
                whatif_children = st.number_input(
                    "What-if Number of Children",
                    min_value=0,
                    max_value=10,
                    value=int(orig["children"]),
                    step=1,
                    key=f"whatif_input_children_{scenario_ver}"
                )

            with w_col5:
                region_options = ["southwest", "southeast", "northwest", "northeast"]
                default_region_idx = region_options.index(orig["region"]) if orig["region"] in region_options else 0
                whatif_region = st.selectbox(
                    "What-if Region",
                    region_options,
                    index=default_region_idx,
                    key=f"whatif_input_region_{scenario_ver}"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            whatif_submitted = st.form_submit_button("Compare Scenario", use_container_width=True)

        if whatif_submitted:
            whatif_df = pd.DataFrame(
                {
                    "age": [whatif_age],
                    "sex": [orig["sex"]],
                    "bmi": [whatif_bmi],
                    "children": [whatif_children],
                    "smoker": [whatif_smoker],
                    "region": [whatif_region]
                }
            )
            whatif_prediction = float(model.predict(whatif_df)[0])
            st.session_state.scenario_prediction = whatif_prediction
            st.session_state.scenario_input = {
                "age": whatif_age,
                "bmi": whatif_bmi,
                "smoker": whatif_smoker,
                "children": whatif_children,
                "region": whatif_region
            }

        if st.session_state.get("scenario_prediction") is not None:
            whatif_pred = st.session_state.scenario_prediction
            scen_input = st.session_state.scenario_input
            diff = whatif_pred - orig_pred
            abs_diff = abs(diff)
            pct_change = (diff / orig_pred * 100.0) if orig_pred > 0 else 0.0

            if diff > 0.01:
                direction_title = "Estimated Cost Increased"
                banner_class = "scenario-banner-increased"
                icon = "📈"
                sign = "+"
                trend_color = "#87445F"
                desc = f"The what-if scenario projects a cost increase of <strong>+${abs_diff:,.2f}</strong> (+{abs(pct_change):.2f}%)."
            elif diff < -0.01:
                direction_title = "Estimated Cost Decreased"
                banner_class = "scenario-banner-decreased"
                icon = "📉"
                sign = "-"
                trend_color = "#126B55"
                desc = f"The what-if scenario projects a cost savings of <strong>-${abs_diff:,.2f}</strong> (-{abs(pct_change):.2f}%)."
            else:
                direction_title = "No Significant Cost Change"
                banner_class = "scenario-banner-neutral"
                icon = "⚖️"
                sign = ""
                trend_color = "#205E53"
                desc = "The what-if scenario projects approximately identical charges to your original estimate ($0.00 / 0.00%)."

            st.markdown(
                f"""<div class="{banner_class}">
                <div style="font-size: 28px; margin-right: 14px; line-height: 1;">{icon}</div>
                <div>
                <div style="font-weight: 700; font-size: 17px; margin-bottom: 3px;">{direction_title}</div>
                <div style="font-size: 14px;">{desc}</div>
                </div>
                </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"""<div class="comparison-card">
                    <div class="comparison-label">Original Cost</div>
                    <div class="comparison-value" style="color: #205E53;">${orig_pred:,.2f}</div>
                    <div class="comparison-caption">{orig['age']} yrs • BMI {orig['bmi']:.1f} • Smoker: {orig['smoker']} • Children: {orig['children']} • {orig['region'].title()}</div>
                    </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(
                    f"""<div class="comparison-card">
                    <div class="comparison-label">What-if Cost</div>
                    <div class="comparison-value" style="color: #126B55;">${whatif_pred:,.2f}</div>
                    <div class="comparison-caption">{scen_input['age']} yrs • BMI {scen_input['bmi']:.1f} • Smoker: {scen_input['smoker']} • Children: {scen_input['children']} • {scen_input['region'].title()}</div>
                    </div>""", unsafe_allow_html=True)

            with c3:
                st.markdown(
                    f"""<div class="comparison-card">
                    <div class="comparison-label">Absolute Difference</div>
                    <div class="comparison-value" style="color: {trend_color};">{sign}${abs_diff:,.2f}</div>
                    <div class="comparison-caption">vs Original Estimate</div>
                    </div>""", unsafe_allow_html=True)

            with c4:
                st.markdown(
                    f"""<div class="comparison-card">
                    <div class="comparison-label">Percentage Change</div>
                    <div class="comparison-value" style="color: {trend_color};">{sign}{abs(pct_change):.2f}%</div>
                    <div class="comparison-caption">relative shift</div>
                    </div>""", unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """<div class="footer">
    MediPredict | Machine Learning Project<br><br>
    This prediction is for educational purposes only and should not be considered professional financial advice.
    </div>""", unsafe_allow_html=True)
