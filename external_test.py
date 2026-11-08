import os
import time
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
from phishing_detector import PhishingDetector


# =========================
# Settings
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "df.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "external_test_predictions.csv")

# Use None to test the whole dataset
# Use a number like 1000 or 5000 for faster stress testing first
SAMPLE_SIZE = None

MODE = "balanced"  # "balanced" or "high_security"


# =========================
# Load dataset
# =========================

df = pd.read_csv(DATA_PATH, low_memory=False)

df.columns = df.columns.str.strip().str.lower()

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

if "label" not in df.columns or "text" not in df.columns:
    raise ValueError("df.csv must contain exactly these columns: label, text")

df["text"] = df["text"].fillna("").astype(str)
df = df[df["text"].str.strip().ne("")].copy()

# =========================
# Standardize labels
# =========================

print("\nOriginal label distribution:")
print(df["label"].value_counts())

# Keep only legitimate and phishing
df = df[df["label"].astype(str).str.strip().isin(["0", "2"])].copy()

df["label_original"] = df["label"]

# Convert to our binary format:
# 0 = legitimate
# 1 = phishing
df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .map({
        "0": 0,
        "2": 1
    })
)

invalid = df[df["label"].isna()]

if len(invalid) > 0:
    print("Invalid labels found:")
    print(invalid["label_original"].value_counts())
    raise ValueError("Fix label mapping before testing.")

df["label"] = df["label"].astype(int)

print("\nBinary label distribution after filtering:")
print(df["label"].value_counts())
print("0 = legitimate, 1 = phishing")

# =========================
# Optional sample
# =========================

if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
    # Stratified sample if possible
    n_per_class = SAMPLE_SIZE // 2

    if df["label"].value_counts().min() >= n_per_class:
        df = (
            df.groupby("label", group_keys=False)
            .sample(n=n_per_class, random_state=42)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
    else:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

print("\nTesting rows:", len(df))
print(df["label"].value_counts())


# =========================
# Load detector
# =========================

detector = PhishingDetector()


# =========================
# Predict
# =========================

predictions = []
prob_legitimate_list = []
prob_phishing_list = []
risk_levels = []

start_time = time.time()

for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"Processed {i}/{len(df)}")

    result = detector.predict_raw_email(
        raw_email=row["text"],
        mode=MODE
    )

    predictions.append(result["predicted_label"])
    prob_legitimate_list.append(result["prob_legitimate"])
    prob_phishing_list.append(result["prob_phishing"])
    risk_levels.append(result["risk_level"])

end_time = time.time()

df["predicted_label"] = predictions
df["prob_legitimate"] = prob_legitimate_list
df["prob_phishing"] = prob_phishing_list
df["risk_level"] = risk_levels

df["prediction_type"] = "correct"

df.loc[
    (df["label"] == 0) & (df["predicted_label"] == 1),
    "prediction_type"
] = "false_positive"

df.loc[
    (df["label"] == 1) & (df["predicted_label"] == 0),
    "prediction_type"
] = "false_negative"


# =========================
# Results
# =========================

y_true = df["label"].values
y_pred = df["predicted_label"].values

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["legitimate", "phishing"],
    digits=4
))

print("\nPrediction types:")
print(df["prediction_type"].value_counts())

print("\nRuntime:")
print(f"{end_time - start_time:.2f} seconds")
print(f"{len(df) / (end_time - start_time):.2f} emails/second")


# =========================
# Save outputs
# =========================

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved predictions to:")
print(OUTPUT_PATH)
