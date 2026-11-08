import os
import time
import csv
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from phishing_detector import PhishingDetector


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = [
    {
        "filename": "AI_legit.csv",
        "source_type": "AI",
        "label": 0
    },
    {
        "filename": "AI_phishing.csv",
        "source_type": "AI",
        "label": 1
    },
    {
        "filename": "H_legit.csv",
        "source_type": "Human",
        "label": 0
    },
    {
        "filename": "H_phishing.csv",
        "source_type": "Human",
        "label": 1
    }
]

OUTPUT_PATH = os.path.join(BASE_DIR, "ai_human_external_predictions.csv")

MODE = "balanced"  # balanced or high_security


def read_csv_safely(path):
    """
    Reads normal CSV files first.
    If parsing fails because text contains unescaped commas,
    fallback reads each line by splitting from the last comma.
    This is useful for files with columns: text,label
    """
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        rows = []

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().lower()

            for line in f:
                line = line.strip()

                if not line:
                    continue

                if "," not in line:
                    continue

                text_part, label_part = line.rsplit(",", 1)

                rows.append({
                    "text": text_part.strip().strip('"'),
                    "label": label_part.strip()
                })

        return pd.DataFrame(rows)


def build_raw_email(row):
    """
    Converts either structured email columns or simple text column
    into the same raw email format used by the website.
    """
    sender = str(row.get("sender", "") or "")
    receiver = str(row.get("receiver", "") or "")
    subject = str(row.get("subject", "") or "")

    if "body" in row and pd.notna(row.get("body", "")):
        body = str(row.get("body", "") or "")
    elif "text" in row and pd.notna(row.get("text", "")):
        body = str(row.get("text", "") or "")
    else:
        body = ""

    parts = []

    if sender.strip():
        parts.append(f"From: {sender.strip()}")

    if receiver.strip():
        parts.append(f"To: {receiver.strip()}")

    if subject.strip():
        parts.append(f"Subject: {subject.strip()}")

    parts.append("")
    parts.append(body.strip())

    return "\n".join(parts).strip()


# =========================
# Load and combine files
# =========================

all_dfs = []

for item in FILES:
    path = os.path.join(BASE_DIR, item["filename"])

    print(f"Loading {item['filename']}...")

    df = read_csv_safely(path)
    df.columns = df.columns.str.strip().str.lower()

    df["source_file"] = item["filename"]
    df["source_type"] = item["source_type"]
    df["label"] = item["label"]

    df["raw_email"] = df.apply(build_raw_email, axis=1)

    df = df[df["raw_email"].str.strip().ne("")].copy()

    all_dfs.append(df[["source_file", "source_type", "label", "raw_email"]])

combined = pd.concat(all_dfs, ignore_index=True)
combined = combined.reset_index(drop=True)

print("\nCombined shape:", combined.shape)

print("\nLabel distribution:")
print(combined["label"].value_counts())

print("\nSource distribution:")
print(pd.crosstab(combined["source_type"], combined["label"]))

print("\nFile distribution:")
print(combined["source_file"].value_counts())


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

for i, row in combined.iterrows():
    if i % 100 == 0:
        print(f"Processed {i}/{len(combined)}")

    result = detector.predict_raw_email(
        raw_email=row["raw_email"],
        mode=MODE
    )

    predictions.append(result["predicted_label"])
    prob_legitimate_list.append(result["prob_legitimate"])
    prob_phishing_list.append(result["prob_phishing"])
    risk_levels.append(result["risk_level"])

end_time = time.time()

combined["predicted_label"] = predictions
combined["prob_legitimate"] = prob_legitimate_list
combined["prob_phishing"] = prob_phishing_list
combined["risk_level"] = risk_levels

combined["prediction_type"] = "correct"

combined.loc[
    (combined["label"] == 0) & (combined["predicted_label"] == 1),
    "prediction_type"
] = "false_positive"

combined.loc[
    (combined["label"] == 1) & (combined["predicted_label"] == 0),
    "prediction_type"
] = "false_negative"


# =========================
# Overall results
# =========================

y_true = combined["label"].values
y_pred = combined["predicted_label"].values

print("\nOverall Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nOverall Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["legitimate", "phishing"],
    digits=4
))

print("\nPrediction types:")
print(combined["prediction_type"].value_counts())


# =========================
# Results by AI vs Human
# =========================

print("\nPrediction types by source type:")
print(pd.crosstab(combined["source_type"], combined["prediction_type"]))

for source_type in combined["source_type"].unique():
    temp = combined[combined["source_type"] == source_type]

    print("\n==============================")
    print(f"Source Type: {source_type}")
    print("==============================")

    print(confusion_matrix(temp["label"], temp["predicted_label"]))

    print(classification_report(
        temp["label"],
        temp["predicted_label"],
        target_names=["legitimate", "phishing"],
        digits=4
    ))


# =========================
# Results by file
# =========================

print("\nPrediction types by file:")
print(pd.crosstab(combined["source_file"], combined["prediction_type"]))


# =========================
# Runtime
# =========================

print("\nRuntime:")
print(f"{end_time - start_time:.2f} seconds")
print(f"{len(combined) / (end_time - start_time):.2f} emails/second")


# =========================
# Save outputs
# =========================

combined.to_csv(OUTPUT_PATH, index=False)

combined[combined["prediction_type"] == "false_positive"].to_csv(
    os.path.join(BASE_DIR, "ai_human_false_positives.csv"),
    index=False
)

combined[combined["prediction_type"] == "false_negative"].to_csv(
    os.path.join(BASE_DIR, "ai_human_false_negatives.csv"),
    index=False
)

print("\nSaved predictions to:")
print(OUTPUT_PATH)