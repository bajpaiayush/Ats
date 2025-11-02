# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load labeled data
print("📂 Loading labeled data...")
df = pd.read_csv("labeled_pairs.csv")

# Combine resume + JD text for feature extraction
df["combined_text"] = df["resume_text"].fillna("") + " " + df["job_description"].fillna("")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"], df["label"], test_size=0.2, random_state=42
)

# Vectorize text
print("🔠 Extracting features...")
vectorizer = TfidfVectorizer(max_features=7000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
print("📊 Evaluating model...")
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.2f}")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump(model, "ats_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("💾 Model and vectorizer saved successfully!")
