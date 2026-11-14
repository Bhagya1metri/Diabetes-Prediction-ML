import altair as alt
import streamlit as st
import pandas as pd
import joblib
import os

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================
# 2. CUSTOM CSS
# =====================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 25px;
        border-radius: 15px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
    }

    .hero p {
        color: #e0f2fe;
        font-size: 16px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e5eaf0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffffff;
        border: 1px solid #dce4ec;
        margin-top: 15px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# =====================================
# 3. LOAD MODEL
# =====================================

@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()


# =====================================
# 3A. LOAD MODEL EVALUATION RESULTS
# =====================================

@st.cache_resource
def load_evaluation():
    evaluation_path = os.path.join(BASE_DIR, "model_evaluation.pkl")
    return joblib.load(evaluation_path)


try:
    evaluation = load_evaluation()
except Exception as e:
    st.error(f"Could not load model_evaluation.pkl: {e}")
    st.info("Run train.py first and make sure model_evaluation.pkl is in the same folder as app.py.")
    st.stop()

# =====================================
# 4. SIDEBAR
# =====================================

with st.sidebar:
    st.title("🩺 Diabetes AI")

    st.write("Machine Learning Dashboard")

    st.divider()

    st.subheader("Project Information")

    st.write("**Algorithm:**")
    st.write("Class-balanced Random Forest")

    st.write("**Dataset:**")
    st.write("Pima Indians Diabetes Dataset")

    st.write("**Features:**")
    st.write("8 health measurements")

    st.write("**Framework:**")
    st.write("Streamlit + Scikit-learn")

    st.divider()

    st.caption(
        "Educational project only. "
        "Predictions are not medical diagnoses."
    )

# =====================================
# 5. HEADER
# =====================================

st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>
        An interactive machine learning application
        powered by a class-balanced Random Forest model.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Enter the measurements below to explore an experimental "
    "machine learning prediction. Use accurate input values."
)

# =====================================
# 6. HEALTH INPUT FORM
# =====================================

st.header("📋 Health Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=1,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=1,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=1,
            max_value=100,
            value=20
        )

    with col2:
        insulin = st.number_input(
            "Insulin Level",
            min_value=1,
            max_value=1000,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=1.0,
            max_value=70.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )

# =====================================
# 7. PREDICTION
# =====================================

if submitted:

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    prediction = model.predict(input_data)[0]

    st.divider()
    st.header("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ Diabetes-positive class")
        st.write(
            "The model classified the entered measurements "
            "into the diabetes-positive class."
        )
    else:
        st.success("✅ Diabetes-negative class")
        st.write(
            "The model classified the entered measurements "
            "into the diabetes-negative class."
        )

    # Model probability, not clinical risk
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        model_score = probabilities[int(prediction)] * 100

        st.metric(
            "Model confidence score",
            f"{model_score:.2f}%"
        )

        st.caption(
            "This is the model's predicted class probability, "
            "not a person's clinical probability of having diabetes."
        )

    st.warning(
        "This is an experimental machine learning prediction, "
        "not a medical diagnosis. A negative prediction does "
        "not rule out diabetes. Consult a qualified healthcare "
        "professional for medical assessment."
    )

    st.subheader("Your Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age"
        ],
        "Value": [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# =====================================
# =====================================
# 8. MODEL PERFORMANCE
# =====================================

st.divider()
st.header("📈 Model Performance")
st.caption("Evaluation results on the independent test dataset.")

balanced_result = next(
    (row for row in evaluation["model_results"]
     if row["Model"] == "Random Forest Balanced"),
    None
)

if balanced_result is None:
    st.error("Balanced Random Forest results were not found in model_evaluation.pkl.")
    st.stop()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f'{balanced_result["Accuracy (%)"]:.2f}%')
m2.metric("Precision", f'{balanced_result["Precision (%)"]:.2f}%')
m3.metric("Positive Recall", f'{balanced_result["Recall (%)"]:.2f}%')
m4.metric("Positive F1-Score", f'{balanced_result["F1-Score (%)"]:.2f}%')
m5.metric("ROC-AUC", f'{balanced_result["ROC-AUC"]:.4f}')

st.info(
    "The balanced Random Forest model has a positive-class recall of "
    f'{balanced_result["Recall (%)"]:.2f}%. Recall measures the proportion '
    "of actual positive cases correctly identified by the model."
)


# =====================================
# 9. MODEL COMPARISON
# =====================================

st.subheader("Model Comparison")

comparison = pd.DataFrame(evaluation["model_results"])
display_columns = [
    "Model", "Accuracy (%)", "Precision (%)",
    "Recall (%)", "F1-Score (%)", "ROC-AUC"
]
comparison = comparison[display_columns]

chart = alt.Chart(comparison).mark_bar(
    cornerRadiusEnd=6,
    color="#1976D2"
).encode(
    x=alt.X(
        "Accuracy (%):Q",
        title="Test Accuracy (%)",
        scale=alt.Scale(domain=[0, 100]),
        axis=alt.Axis(tickCount=10)
    ),
    y=alt.Y(
        "Model:N",
        title=None,
        sort="-x",
        axis=alt.Axis(labelLimit=300)
    ),
    tooltip=[
        alt.Tooltip("Model:N"),
        alt.Tooltip("Accuracy (%):Q", format=".2f"),
        alt.Tooltip("Precision (%):Q", format=".2f"),
        alt.Tooltip("Recall (%):Q", format=".2f"),
        alt.Tooltip("F1-Score (%):Q", format=".2f"),
        alt.Tooltip("ROC-AUC:Q", format=".4f")
    ]
).properties(height=260)

st.altair_chart(chart, use_container_width=True)
st.dataframe(comparison, use_container_width=True, hide_index=True)


# =====================================
# 9A. ROC CURVE
# =====================================

st.divider()
st.subheader("ROC Curve Comparison")

roc_records = []
for model_name, values in evaluation["roc_data"].items():
    for fpr_value, tpr_value in zip(values["fpr"], values["tpr"]):
        roc_records.append({
            "Model": model_name,
            "False Positive Rate": float(fpr_value),
            "True Positive Rate": float(tpr_value),
            "ROC-AUC": float(values["auc"])
        })

roc_df = pd.DataFrame(roc_records)

roc_chart = alt.Chart(roc_df).mark_line(strokeWidth=2).encode(
    x=alt.X(
        "False Positive Rate:Q",
        scale=alt.Scale(domain=[0, 1]),
        title="False Positive Rate"
    ),
    y=alt.Y(
        "True Positive Rate:Q",
        scale=alt.Scale(domain=[0, 1]),
        title="True Positive Rate"
    ),
    color=alt.Color("Model:N", title="Model"),
    tooltip=[
        alt.Tooltip("Model:N"),
        alt.Tooltip("False Positive Rate:Q", format=".3f"),
        alt.Tooltip("True Positive Rate:Q", format=".3f"),
        alt.Tooltip("ROC-AUC:Q", format=".4f")
    ]
).properties(height=380)

reference_df = pd.DataFrame({
    "False Positive Rate": [0, 1],
    "True Positive Rate": [0, 1]
})
reference_line = alt.Chart(reference_df).mark_line(
    strokeDash=[6, 5],
    color="gray"
).encode(
    x="False Positive Rate:Q",
    y="True Positive Rate:Q"
)

st.altair_chart(roc_chart + reference_line, use_container_width=True)
st.caption(
    "A curve closer to the top-left corner generally indicates better "
    "ability to distinguish between the two classes. ROC-AUC summarizes "
    "the curve; it is not a measure of clinical accuracy."
)


# =====================================
# 9B. CLASSIFICATION REPORT
# =====================================

st.subheader("Classification Report — Balanced Random Forest")
report_df = pd.DataFrame(evaluation["classification_report"]).transpose()
st.dataframe(report_df.round(3), use_container_width=True)


# 10. CONFUSION MATRIX
# =====================================

st.divider()
st.header("🎯 Confusion Matrix")

matrix_path = os.path.join(
    BASE_DIR,
    "confusion_matrix.png"
)

if os.path.exists(matrix_path):
    st.image(
        matrix_path,
        caption="Balanced Random Forest — Test Dataset",
        use_container_width=True
    )
else:
    st.warning("Confusion matrix image not found.")

# =====================================
# 11. FEATURE IMPORTANCE
# =====================================

st.divider()
st.header("🔬 Feature Importance")

classifier = model.named_steps["classifier"]

feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_chart = alt.Chart(importance_df).mark_bar(
    cornerRadiusEnd=6,
    color="#1976D2"
).encode(
    x=alt.X(
        "Importance:Q",
        title="Feature Importance"
    ),
    y=alt.Y(
        "Feature:N",
        title=None,
        sort="-x",
        axis=alt.Axis(labelLimit=300)
    ),
    tooltip=[
        alt.Tooltip("Feature:N"),
        alt.Tooltip("Importance:Q", format=".4f")
    ]
).properties(
    height=350
)

st.altair_chart(
    importance_chart,
    use_container_width=True
)

st.caption(
    "Feature importance shows how much each feature "
    "contributed to the Random Forest's split decisions. "
    "It does not establish medical causation."
)

# =====================================
# 12. FOOTER
# =====================================

st.divider()

st.caption(
    "Diabetes Prediction AI | Built with Python, "
    "Pandas, Scikit-learn and Streamlit | "
    "Educational use only — not medical advice."
)
