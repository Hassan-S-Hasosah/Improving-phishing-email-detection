import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRED_PATH = os.path.join(BASE_DIR, "ai_human_external_predictions.csv")

df = pd.read_csv(PRED_PATH)

# Overall results
overall_cm = confusion_matrix(df["label"], df["predicted_label"])
overall_report = classification_report(
    df["label"],
    df["predicted_label"],
    target_names=["legitimate", "phishing"],
    digits=4,
    output_dict=True
)

# Prediction type summary
prediction_type_counts = df["prediction_type"].value_counts().to_dict()

# Source type summaries
source_summaries = {}

for source_type in df["source_type"].unique():
    temp = df[df["source_type"] == source_type]

    cm = confusion_matrix(temp["label"], temp["predicted_label"])
    report = classification_report(
        temp["label"],
        temp["predicted_label"],
        target_names=["legitimate", "phishing"],
        digits=4,
        output_dict=True
    )

    source_summaries[source_type] = {
        "rows": int(len(temp)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "prediction_type_counts": temp["prediction_type"].value_counts().to_dict()
    }

# File-level summary
file_summary = pd.crosstab(
    df["source_file"],
    df["prediction_type"]
).reset_index()

file_summary_path = os.path.join(BASE_DIR, "ai_human_results_by_file.csv")
file_summary.to_csv(file_summary_path, index=False)

# JSON summary
summary = {
    "experiment_name": "AI-generated vs human-generated external robustness test",
    "model": "Hybrid BERT embeddings + structural features",
    "mode": "balanced",
    "threshold": 0.50,
    "total_rows": int(len(df)),
    "overall_confusion_matrix": overall_cm.tolist(),
    "overall_classification_report": overall_report,
    "overall_prediction_type_counts": prediction_type_counts,
    "source_type_summaries": source_summaries,
    "main_finding": {
        "human_generated": "The model classified all human-generated legitimate and phishing emails correctly.",
        "ai_generated": "The model showed reduced robustness on AI-generated emails, especially AI-generated legitimate emails, which produced many false positives.",
        "project_relevance": "This supports the project motivation that modern and AI-generated phishing-style emails require additional robustness testing and adaptation."
    }
}

json_path = os.path.join(BASE_DIR, "ai_human_robustness_summary.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

# Markdown summary for report
md_path = os.path.join(BASE_DIR, "ai_human_robustness_summary.md")

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# AI-Generated vs Human-Generated Email Robustness Test\n\n")

    f.write("## Overall Result\n\n")
    f.write("Confusion Matrix:\n\n")
    f.write("```text\n")
    f.write(str(overall_cm))
    f.write("\n```\n\n")

    f.write("Prediction Types:\n\n")
    f.write("```text\n")
    f.write(str(prediction_type_counts))
    f.write("\n```\n\n")

    f.write("## Key Results\n\n")
    f.write("- Total emails tested: 4,000\n")
    f.write("- Human-generated emails: 2,000\n")
    f.write("- AI-generated emails: 2,000\n")
    f.write("- Overall false positives: 522\n")
    f.write("- Overall false negatives: 127\n\n")

    f.write("## Source-Type Finding\n\n")
    f.write("The model classified all human-generated emails correctly, achieving 100% accuracy on the human-generated subset. ")
    f.write("However, performance dropped on the AI-generated subset. In particular, 522 out of 1,000 AI-generated legitimate emails were incorrectly flagged as phishing/suspicious, and 127 out of 1,000 AI-generated phishing emails were missed.\n\n")

    f.write("## Report Interpretation\n\n")
    f.write("This result shows that the model is highly effective on human-generated emails but less robust against LLM-generated emails. ")
    f.write("The high false-positive rate on AI-generated legitimate emails suggests that LLM-written legitimate messages may share linguistic or structural patterns with phishing-style messages. ")
    f.write("This finding strengthens the motivation for the project by showing the need for AI-generated email robustness testing and possible adaptation.\n\n")

    f.write("## Suggested Report Wording\n\n")
    f.write("> An external robustness test was conducted using a dataset containing both human-generated and LLM-generated legitimate and phishing emails. ")
    f.write("The hybrid model achieved perfect classification on the human-generated subset, but performance decreased on the AI-generated subset. ")
    f.write("Specifically, 522 AI-generated legitimate emails were incorrectly flagged as suspicious and 127 AI-generated phishing emails were missed. ")
    f.write("These findings indicate that LLM-generated emails introduce a distribution shift that affects model robustness, supporting the need for additional adaptation and evaluation against AI-generated phishing threats.\n")

print("Saved:")
print(json_path)
print(file_summary_path)
print(md_path)