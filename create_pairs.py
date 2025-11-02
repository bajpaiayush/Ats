# create_pairs.py
import pandas as pd
import random

# Files (put them in same folder)
RESUMES_FILE = "final_merged_resumes.csv"   # or merged_resumes.csv
JDS_FILE = "job_descriptions.csv"
OUT_FILE = "resume_jd_pairs.csv"

# How many JD samples per resume (reduce for less data)
K = 3

print("Loading data...")
resumes = pd.read_csv(RESUMES_FILE)
jds = pd.read_csv(JDS_FILE)

pairs = []
for _, r in resumes.iterrows():
    chosen = jds.sample(n=min(K, len(jds)), random_state=random.randint(1,10000))
    for _, jd in chosen.iterrows():
        pairs.append({
            "person_id": r.get("person_id", ""),
            "resume_text": r.get("Full_Resume_Text", ""),
            "company": jd["company"],
            "role": jd["role"],
            "job_description": jd["job_description"]
        })

pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv(OUT_FILE, index=False)
print(f"Saved {len(pairs_df)} pairs to {OUT_FILE}")
