# %%
import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "hybrid_full_outputs")

test_pred_path = os.path.join(OUTPUT_DIR, "hybrid_full_test_predictions.csv")

test_df = pd.read_csv(test_pred_path)

print(test_df.shape)
print(test_df.columns)
print(test_df["prediction_type"].value_counts())

# %%
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


y_true = test_df["true_label"].values
probs_phishing = test_df["prob_phishing"].values

thresholds = np.arange(0.10, 0.91, 0.01)

results = [
    evaluate_threshold(y_true, probs_phishing, threshold)
    for threshold in thresholds
]

threshold_results = pd.DataFrame(results)

print(threshold_results.sort_values(["fn", "f1"], ascending=[True, False]).head(20))

# %%
threshold_results.to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_full_threshold_tuning_test_only.csv"),
    index=False
)

print("Saved threshold tuning table.")
# %%
good_thresholds = threshold_results[
    (threshold_results["fn"] < 98) &
    (threshold_results["fp"] <= 150)
].sort_values(["fn", "f1"], ascending=[True, False])

print(good_thresholds.head(30))
# %%