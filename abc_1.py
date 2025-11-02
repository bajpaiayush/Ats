import pandas as pd

# ---- Load the already merged file ----
df = pd.read_csv("merged_resumes.csv")

# ---- Fill missing values ----
df = df.fillna("")

# ---- Detect text-based columns dynamically ----
possible_text_cols = [
    col for col in df.columns
    if any(x in col.lower() for x in [
        "ability", "education", "experience", "skill", "title", "firm", "program", "institution"
    ])
]

print("✅ Text columns detected for resume creation:", possible_text_cols)

# ---- Create combined resume text ----
df["Full_Resume_Text"] = df[possible_text_cols].apply(lambda x: " ".join(x.astype(str)), axis=1)

# ---- Save final cleaned file ----
df.to_csv("final_merged_resumes.csv", index=False)

# ---- Summary ----
print("\n✅ Fixed file saved as final_merged_resumes.csv")
print("📊 Rows:", len(df))
print("📋 Columns:", df.columns.tolist())
