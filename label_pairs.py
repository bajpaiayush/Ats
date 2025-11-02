# label_pairs.py (memory-optimized version)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

IN_FILE = "resume_jd_pairs.csv"
OUT_FILE = "labeled_pairs.csv"
THRESHOLD = 0.01  # tune this: 0.5–0.65 typical

print("📂 Loading pairs...")
df = pd.read_csv(IN_FILE)

# Clean text
resumes = df["resume_text"].fillna("").tolist()
jobs = df["job_description"].fillna("").tolist()

print("🔠 Fitting TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
vectorizer.fit(resumes + jobs)

resume_vecs = vectorizer.transform(resumes)
job_vecs = vectorizer.transform(jobs)

print("⚙️ Computing cosine similarities in batches...")
batch_size = 1000
sims = np.zeros(len(df))

for start in range(0, len(df), batch_size):
    end = min(start + batch_size, len(df))
    batch_resume_vecs = resume_vecs[start:end]
    batch_job_vecs = job_vecs[start:end]

    # Compare row i of resumes with row i of jobs
    batch_sims = np.array([
        cosine_similarity(batch_resume_vecs[i], batch_job_vecs[i])[0][0]
        for i in range(end - start)
    ])
    sims[start:end] = batch_sims

print("🏷️ Labeling pairs...")
labels = (sims >= THRESHOLD).astype(int)

df["similarity"] = sims
df["label"] = labels

df.to_csv(OUT_FILE, index=False)
print(f"✅ Saved labeled pairs to {OUT_FILE} (threshold={THRESHOLD})")
