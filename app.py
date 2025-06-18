
import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Page config and custom CSS
st.set_page_config(page_title="Quick Diagnosis", layout="wide")
st.markdown("""
<style>
body, .stApp {
    background-color: #111;
    color: #ddd;
}
input, select {
    background-color: #222 !important;
    color: #eee !important;
    border: 1px solid #444 !important;
}
.stNumberInput, .stSelectbox {
    margin-bottom: 12px !important;
}
.stButton>button {
    background: #444 !important;
    color: #fff !important;
    border-radius: 6px;
    height: 40px;
    font-size: 15px;
}
hr {
    border: 1px solid #444;
}
.prediction-badge {
    background-color: #2a2a2a;
    padding: 14px 24px;
    border-radius: 12px;
    display: inline-block;
    margin-top: 16px;
    margin-bottom: 16px;
}
.prediction-badge span.label {
    color: #eee;
    font-size: 22px;
    font-weight: bold;
    margin-right: 10px;
}
.prediction-badge span.value {
    color: #49f2a5;
    font-size: 20px;
    font-weight: 600;
}
.block-spacer {
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Quick Diagnosis")

left, right = st.columns([1.2, 1.1], gap="large")

with left:
    st.subheader("Fill Patient Info")
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 0, 120, 30)
            bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
            gender = st.selectbox("Gender", encoders['Gender'].classes_)
        with col2:
            bp = st.number_input("BP", 80, 200, 120)
            sugar = st.number_input("Sugar", 50, 300, 100)
            smoking = st.selectbox("Smoking", encoders['Smoking'].classes_)
        with col3:
            cholesterol = st.number_input("Cholesterol", 100, 400, 180)
            fam = st.selectbox("Family History", encoders['FamilyHistory'].classes_)

        st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)
        submit = st.form_submit_button("🔍 Predict")

if submit:
    X = pd.DataFrame([{
        "Age": age,
        "Gender": encoders['Gender'].transform([gender])[0],
        "BMI": bmi,
        "BP": bp,
        "Sugar": sugar,
        "Cholesterol": cholesterol,
        "Smoking": encoders['Smoking'].transform([smoking])[0],
        "FamilyHistory": encoders['FamilyHistory'].transform([fam])[0],
    }])

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    classes = encoders['Disease'].inverse_transform(model.classes_)

    top_idx = probs.argmax()
    top_class = classes[top_idx]
    top_prob = probs[top_idx] * 100

    with right:
        st.subheader("🧠 Result & Advice")

        # Aligned badge
        st.markdown(f"""
        <div class="prediction-badge">
            <span class="label">🎯 Prediction:</span><span class="value">{top_class}</span>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        st.markdown(f"**Confidence**: `{top_prob:.2f}%`")

        # Probability breakdown
        with st.expander("📊 Probability Breakdown", expanded=False):
            for cls, prob in zip(classes, probs):
                st.markdown(f"- **{cls}**: `{prob*100:.2f}%`")

        # Advice
        st.markdown('<div class="block-spacer"></div>', unsafe_allow_html=True)
        st.markdown("### 💡 Recommendation")
        if top_class == "No Disease":
            st.success("You're in the clear! Keep maintaining a healthy lifestyle 💪")
        else:
            st.warning("Consult a doctor for further diagnosis.")
            st.markdown("""
            - 🥗 Maintain a balanced diet  
            - 🏃 Exercise regularly  
            - 🚭 Avoid smoking and alcohol  
            - 🩸 Monitor sugar and BP regularly  
            """)
