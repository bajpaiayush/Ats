import pandas as pd

df = pd.read_csv("final_merged_resumes.csv")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print(df.head(3))
