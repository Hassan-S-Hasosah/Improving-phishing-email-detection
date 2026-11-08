import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support


# =========================
# 1. Paths and settings
# =========================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_DIR = os.path.join(PROJECT_DIR, "bert_full_model")

TRAIN_PATH = os.path.join(PROJECT_DIR, "train.csv")
VAL_PATH = os.path.join(PROJECT_DIR, "validation.csv")
TEST_PATH = os.path.join(PROJECT_DIR, "test.csv")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "hybrid_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH = 256
BERT_BATCH_SIZE = 16   # If CUDA out of memory, change to 8
MLP_BATCH_SIZE = 256
RANDOM_STATE = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 2. Load data
# =========================

train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
val_df = pd.read_csv(VAL_PATH, low_memory=False)
test_df = pd.read_csv(TEST_PATH, low_memory=False)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# Medium sample first
train_medium = (
    train_df.groupby("label", group_keys=False)
    .sample(n=25000, random_state=RANDOM_STATE)
)

val_medium = (
    val_df.groupby("label", group_keys=False)
    .sample(n=5000, random_state=RANDOM_STATE)
)

print("\nTrain medium labels:")
print(train_medium["label"].value_counts())

print("\nValidation medium labels:")
print(val_medium["label"].value_counts())


# =========================
# 3. Structural features
# =========================

structural_features = [
    "url_present",
    "url_count",
    "subject_length",
    "body_length",
    "text_length",
    "subject_word_count",
    "body_word_count",
    "has_sender",
    "has_receiver",
    "exclamation_count",
    "question_count"
]

binary_features = [
    "url_present",
    "has_sender",
    "has_receiver"
]

numeric_features = [
    "url_count",
    "subject_length",
    "body_length",
    "text_length",
    "subject_word_count",
    "body_word_count",
    "exclamation_count",
    "question_count"
]


def prepare_structural_features(df, scaler=None, fit_scaler=False):
    features_df = df[structural_features].copy()

    for col in structural_features:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

    # Log-transform skewed count/length values
    features_df[numeric_features] = np.log1p(features_df[numeric_features])

    if fit_scaler:
        scaler = StandardScaler()
        features_df[numeric_features] = scaler.fit_transform(features_df[numeric_features])
    else:
        features_df[numeric_features] = scaler.transform(features_df[numeric_features])

    features_df[binary_features] = features_df[binary_features].astype(float)

    return features_df.values.astype(np.float32), scaler


X_train_struct, scaler = prepare_structural_features(train_medium, fit_scaler=True)
X_val_struct, _ = prepare_structural_features(val_medium, scaler=scaler, fit_scaler=False)
X_test_struct, _ = prepare_structural_features(test_df, scaler=scaler, fit_scaler=False)

joblib.dump(scaler, os.path.join(OUTPUT_DIR, "hybrid_structural_scaler.pkl"))

print("\nStructural feature shapes:")
print(X_train_struct.shape)
print(X_val_struct.shape)
print(X_test_struct.shape)


# =========================
# 4. Load saved BERT
# =========================

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
bert_classifier = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)

# Use the BERT encoder part only
bert_model = bert_classifier.bert
bert_model.to(device)
bert_model.eval()

print("\nBERT loaded.")


# =========================
# 5. Extract BERT embeddings
# =========================

def extract_bert_embeddings(texts, split_name):
    save_path = os.path.join(OUTPUT_DIR, f"{split_name}_bert_embeddings.npy")

    if os.path.exists(save_path):
        print(f"Loading existing embeddings: {save_path}")
        return np.load(save_path)

    all_embeddings = []

    texts = texts.fillna("").astype(str).tolist()

    print(f"\nExtracting BERT embeddings for {split_name}: {len(texts)} rows")

    for start in range(0, len(texts), BERT_BATCH_SIZE):
        batch_texts = texts[start:start + BERT_BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = bert_model(**inputs)
            else:
                outputs = bert_model(**inputs)

            if outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(embeddings.float().cpu().numpy())

        if (start // BERT_BATCH_SIZE) % 100 == 0:
            print(f"{split_name}: processed {start}/{len(texts)}")

    embeddings_array = np.vstack(all_embeddings).astype(np.float32)
    np.save(save_path, embeddings_array)

    print(f"Saved embeddings to: {save_path}")
    print(f"{split_name} embeddings shape:", embeddings_array.shape)

    return embeddings_array


X_train_bert = extract_bert_embeddings(train_medium["text"], "train_medium")
X_val_bert = extract_bert_embeddings(val_medium["text"], "val_medium")
X_test_bert = extract_bert_embeddings(test_df["text"], "test_full")


# =========================
# 6. Combine BERT + structural features
# =========================

X_train = np.hstack([X_train_bert, X_train_struct]).astype(np.float32)
X_val = np.hstack([X_val_bert, X_val_struct]).astype(np.float32)
X_test = np.hstack([X_test_bert, X_test_struct]).astype(np.float32)

y_train = train_medium["label"].astype(int).values
y_val = val_medium["label"].astype(int).values
y_test = test_df["label"].astype(int).values

print("\nHybrid feature shapes:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


# =========================
# 7. Build MLP hybrid classifier
# =========================

class HybridMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


train_loader = make_loader(X_train, y_train, MLP_BATCH_SIZE, shuffle=True)
val_loader = make_loader(X_val, y_val, MLP_BATCH_SIZE, shuffle=False)
test_loader = make_loader(X_test, y_test, MLP_BATCH_SIZE, shuffle=False)

model = HybridMLP(input_dim=X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)


# =========================
# 8. Train MLP
# =========================

def evaluate_model(model, loader):
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    acc = accuracy_score(all_labels, all_preds)

    return acc, precision, recall, f1, all_labels, all_preds, all_probs


best_val_f1 = 0
best_model_path = os.path.join(OUTPUT_DIR, "hybrid_bert_mlp_medium.pt")

EPOCHS = 20

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_acc, val_precision, val_recall, val_f1, _, _, _ = evaluate_model(model, val_loader)

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Precision: {val_precision:.4f} | "
        f"Val Recall: {val_recall:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        print("Saved best model.")


# =========================
# 9. Evaluate best model on validation and test
# =========================

model.load_state_dict(torch.load(best_model_path, map_location=device))

val_acc, val_precision, val_recall, val_f1, y_val_true, y_val_pred, y_val_probs = evaluate_model(model, val_loader)

print("\nFinal Validation Confusion Matrix:")
print(confusion_matrix(y_val_true, y_val_pred))

print("\nFinal Validation Classification Report:")
print(classification_report(
    y_val_true,
    y_val_pred,
    target_names=["legitimate", "phishing"],
    digits=4
))

test_acc, test_precision, test_recall, test_f1, y_test_true, y_test_pred, y_test_probs = evaluate_model(model, test_loader)

print("\nFinal Test Confusion Matrix:")
print(confusion_matrix(y_test_true, y_test_pred))

print("\nFinal Test Classification Report:")
print(classification_report(
    y_test_true,
    y_test_pred,
    target_names=["legitimate", "phishing"],
    digits=4
))


# =========================
# 10. Save hybrid predictions
# =========================

test_results_df = test_df.copy()
test_results_df["true_label"] = y_test_true
test_results_df["predicted_label"] = y_test_pred
test_results_df["prob_legitimate"] = y_test_probs[:, 0]
test_results_df["prob_phishing"] = y_test_probs[:, 1]

test_results_df["prediction_type"] = "correct"

test_results_df.loc[
    (test_results_df["true_label"] == 0) & (test_results_df["predicted_label"] == 1),
    "prediction_type"
] = "false_positive"

test_results_df.loc[
    (test_results_df["true_label"] == 1) & (test_results_df["predicted_label"] == 0),
    "prediction_type"
] = "false_negative"

test_results_df.to_csv(os.path.join(OUTPUT_DIR, "hybrid_medium_test_predictions.csv"), index=False)
test_results_df[test_results_df["prediction_type"] != "correct"].to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_medium_misclassified.csv"),
    index=False
)
test_results_df[test_results_df["prediction_type"] == "false_negative"].to_csv(
    os.path.join(OUTPUT_DIR, "hybrid_medium_false_negatives.csv"),
    index=False
)

print("\nSaved hybrid outputs to:", OUTPUT_DIR)
##################################################

import json
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test_true, y_test_pred)

report_dict = classification_report(
    y_test_true,
    y_test_pred,
    target_names=["legitimate", "phishing"],
    digits=4,
    output_dict=True
)

metrics_summary = {
    "model": "hybrid_bert_embeddings_structural_mlp",
    "training_type": "medium_train_sample",
    "bert_base": "bert_full_model",
    "train_rows": len(train_medium),
    "validation_rows": len(val_medium),
    "test_rows": len(test_df),
    "structural_features": structural_features,
    "confusion_matrix": cm.tolist(),
    "classification_report": report_dict,
    "false_negatives": int(((y_test_true == 1) & (y_test_pred == 0)).sum()),
    "false_positives": int(((y_test_true == 0) & (y_test_pred == 1)).sum())
}

with open(os.path.join(OUTPUT_DIR, "hybrid_medium_metrics_summary.json"), "w") as f:
    json.dump(metrics_summary, f, indent=4)

print("Saved hybrid medium metrics summary.")