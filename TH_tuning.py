import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

df = pd.read_csv("external_test_predictions.csv")

def evaluate_threshold(y_true, probs_phishing, threshold):
    y_pred = (probs_phishing >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

y_true = df["label"].values
probs_phishing = df["prob_phishing"].values

thresholds = np.arange(0.50, 0.991, 0.01)

results = [
    evaluate_threshold(y_true, probs_phishing, t)
    for t in thresholds
]

threshold_df = pd.DataFrame(results)

# Show thresholds that reduce false positives while keeping phishing recall high
filtered = threshold_df[
    (threshold_df["recall"] >= 0.97)
].sort_values(["fp", "fn"], ascending=[True, True])

print(filtered.head(30).to_string(index=False))

threshold_df.to_csv("external_full_threshold_tuning.csv", index=False)