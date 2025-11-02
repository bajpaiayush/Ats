import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load model and vectorizer
model = joblib.load("ats_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="ATS Resume Checker", layout="centered")

# Add background image using CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1522205408450-add114ad53fe");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    border-radius: 10px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title
st.title("💼 AI-Powered ATS Resume Checker")
st.markdown("### Check how well your resume matches a job description!")

# Input boxes
jd = st.text_area("📄 Paste the Job Description here:")
resume = st.text_area("🧾 Paste your Resume here:")

if st.button("Check Match"):
    if jd.strip() == "" or resume.strip() == "":
        st.warning("⚠️ Please enter both Job Description and Resume.")
    else:
        # Combine and vectorize
        texts = [jd, resume]
        vectors = vectorizer.transform(texts)
        sim = np.dot(vectors[0].toarray(), vectors[1].toarray().T)[0][0]
        sim_score = round(sim * 100, 2)

        # Predict match using model
        features = vectorizer.transform([resume])
        prediction = model.predict(features)[0]

        # Output section
        st.subheader("🔍 Match Result:")
        st.progress(sim_score / 100)
        st.success(f"✅ Your Resume Match Score: **{sim_score}%**")

        if prediction == 1:
            st.markdown("### 🎯 This resume is a **Good Match!**")
        else:
            st.markdown("### ⚙️ This resume could be **Improved** for better match.")

st.markdown("---")
st.caption("Made with ❤️ by Team ATS")

