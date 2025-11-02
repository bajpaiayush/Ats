import pandas as pd

df = pd.read_csv("labeled_pairs.csv")
print(df["label"].value_counts())
print("Average similarity:", df["similarity"].mean())
