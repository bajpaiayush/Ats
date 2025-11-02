import streamlit as st
import joblib
import PyPDF2
import io

# Load model and vectorizer
model = joblib.load("ats_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("AI ATS Resume Checker")

# Upload resume instead of paste
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text()
else:
    resume_text = ""

# Upload job description
job_description = st.text_area("Paste the job description here")

# Predict similarity
if st.button("Analyze"):
    if uploaded_file is not None and job_description.strip():
        features = vectorizer.transform([resume_text, job_description])
       from scipy.spatial.distance import cosine

# Compute similarity manually
similarity = 1 - cosine(features[0].toarray(), features[1].toarray())

        st.success(f"Similarity score: {similarity}")
    else:
        st.warning("Please upload a resume and enter a job description.")

