import os
import json
import re
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# Paths
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BERT_MODEL_DIR = os.path.join(BASE_DIR, "bert_full_model")
OLD_HYBRID_DIR = os.path.join(BASE_DIR, "hybrid_full_outputs")
NEW_OUTPUT_DIR = os.path.join(BASE_DIR, "hybrid_ai_adapted_outputs")
os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

AI_LEGIT_PATH = os.path.join(BASE_DIR, "AI_legit.csv")
AI_PHISH_PATH = os.path.join(BASE_DIR, "AI_phishing.csv")

OLD_MLP_PATH = os.path.join(OLD_HYBRID_DIR, "hybrid_bert_mlp_full.pt")

TRAIN_EMB_PATH = os.path.join(OLD_HYBRID_DIR, "train_full_bert_embeddings.npy")
VAL_EMB_PATH = os.path.join(OLD_HYBRID_DIR, "val_full_bert_embeddings.npy")
TEST_EMB_PATH = os.path.join(OLD_HYBRID_DIR, "test_full_bert_embeddings.npy")

MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
MLP_BATCH_SIZE = 512
EPOCHS = 15
PATIENCE = 4
AI_REPEAT = 15
RANDOM_STATE = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# Features
# =========================

STRUCTURAL_FEATURES = [
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

BINARY_FEATURES = [
    "url_present",
    "has_sender",
    "has_receiver"
]

NUMERIC_FEATURES = [
    "url_count",
    "subject_length",
    "body_length",
    "text_length",
    "subject_word_count",
    "body_word_count",
    "exclamation_count",
    "question_count"
]


# =========================
# Model
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


def safe_load_state_dict(path):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================
# Helpers
# =========================

def read_csv_safely(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "," in line:
                    text_part, label_part = line.rsplit(",", 1)
                    rows.append({"text": text_part.strip().strip('"'), "label": label_part.strip()})
                else:
                    rows.append({"text": line, "label": ""})
        return pd.DataFrame(rows)


def extract_urls(text):
    if not text:
        return []
    return re.findall(r"(https?://\S+|www\.\S+)", text)


def parse_raw_email(raw_email):
    raw_email = raw_email or ""

    subject = ""
    sender = ""
    receiver = ""

    lines = raw_email.splitlines()
    body_lines = []

    for line in lines:
        clean = line.strip()
        lower = clean.lower()

        if lower.startswith("subject:") and subject == "":
            subject = clean.split(":", 1)[1].strip()
        elif lower.startswith("from:") and sender == "":
            sender = clean.split(":", 1)[1].strip()
        elif lower.startswith("to:") and receiver == "":
            receiver = clean.split(":", 1)[1].strip()
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()

    if body == "":
        body = raw_email.strip()

    return subject, body, sender, receiver


def compute_structural_from_raw(raw_email):
    subject, body, sender, receiver = parse_raw_email(raw_email)

    text = f"Subject: {subject}\nBody: {body}".strip()
    urls = extract_urls(subject + " " + body)

    return {
        "url_present": 1 if len(urls) > 0 else 0,
        "url_count": len(urls),
        "subject_length": len(subject),
        "body_length": len(body),
        "text_length": len(text),
        "subject_word_count": len(subject.split()),
        "body_word_count": len(body.split()),
        "has_sender": 1 if sender.strip() else 0,
        "has_receiver": 1 if receiver.strip() else 0,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?")
    }


def build_raw_email_from_row(row):
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


def prepare_structural_matrix(df, scaler=None, fit_scaler=False):
    features_df = df[STRUCTURAL_FEATURES].copy()

    for col in STRUCTURAL_FEATURES:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

    features_df[NUMERIC_FEATURES] = np.log1p(features_df[NUMERIC_FEATURES])

    if fit_scaler:
        scaler = StandardScaler()
        features_df[NUMERIC_FEATURES] = scaler.fit_transform(features_df[NUMERIC_FEATURES])
    else:
        features_df[NUMERIC_FEATURES] = scaler.transform(features_df[NUMERIC_FEATURES])

    features_df[BINARY_FEATURES] = features_df[BINARY_FEATURES].astype(float)

    return features_df.values.astype(np.float32), scaler


def evaluate_model(model, X, y, batch_size=512):
    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="binary",
        pos_label=1,
        zero_division=0
    )

    acc = accuracy_score(all_labels, all_preds)

    return acc, precision, recall, f1, all_labels, all_preds, all_probs


def print_report(title, y_true, y_pred):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["legitimate", "phishing"],
        digits=4
    ))


# =========================
# Load original data and embeddings
# =========================

print("\nLoading original GP2 data...")
train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
val_df = pd.read_csv(VAL_PATH, low_memory=False)
test_df = pd.read_csv(TEST_PATH, low_memory=False)

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print("\nLoading existing BERT embeddings...")
X_train_bert = np.load(TRAIN_EMB_PATH)
X_val_bert = np.load(VAL_EMB_PATH)
X_test_bert = np.load(TEST_EMB_PATH)

print("Train embeddings:", X_train_bert.shape)
print("Val embeddings:", X_val_bert.shape)
print("Test embeddings:", X_test_bert.shape)


# =========================
# Load AI data
# =========================

print("\nLoading AI-generated datasets...")

ai_legit = read_csv_safely(AI_LEGIT_PATH)
ai_phish = read_csv_safely(AI_PHISH_PATH)

ai_legit.columns = ai_legit.columns.str.strip().str.lower()
ai_phish.columns = ai_phish.columns.str.strip().str.lower()

ai_legit["label"] = 0
ai_phish["label"] = 1

ai_legit["source_file"] = "AI_legit.csv"
ai_phish["source_file"] = "AI_phishing.csv"

ai_legit["raw_email"] = ai_legit.apply(build_raw_email_from_row, axis=1)
ai_phish["raw_email"] = ai_phish.apply(build_raw_email_from_row, axis=1)

ai_df = pd.concat([
    ai_legit[["raw_email", "label", "source_file"]],
    ai_phish[["raw_email", "label", "source_file"]]
], ignore_index=True)

ai_df = ai_df[ai_df["raw_email"].str.strip().ne("")].copy()
ai_df = ai_df.reset_index(drop=True)

print("AI dataset:", ai_df.shape)
print(ai_df["label"].value_counts())


# =========================
# Split AI data
# =========================

ai_train, ai_temp = train_test_split(
    ai_df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=ai_df["label"]
)

ai_val, ai_test = train_test_split(
    ai_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=ai_temp["label"]
)

ai_train = ai_train.reset_index(drop=True)
ai_val = ai_val.reset_index(drop=True)
ai_test = ai_test.reset_index(drop=True)

ai_train.to_csv(os.path.join(NEW_OUTPUT_DIR, "ai_train_split.csv"), index=False)
ai_val.to_csv(os.path.join(NEW_OUTPUT_DIR, "ai_val_split.csv"), index=False)
ai_test.to_csv(os.path.join(NEW_OUTPUT_DIR, "ai_test_split.csv"), index=False)

print("\nAI splits:")
print("AI train:", ai_train.shape, ai_train["label"].value_counts().to_dict())
print("AI val:", ai_val.shape, ai_val["label"].value_counts().to_dict())
print("AI test:", ai_test.shape, ai_test["label"].value_counts().to_dict())


# =========================
# Load BERT for AI embeddings
# =========================

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
bert_classifier = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
bert_model = bert_classifier.bert
bert_model.to(device)
bert_model.eval()


def extract_ai_embeddings(texts, split_name):
    save_path = os.path.join(NEW_OUTPUT_DIR, f"{split_name}_bert_embeddings.npy")

    if os.path.exists(save_path):
        print(f"Loading existing embeddings: {save_path}")
        return np.load(save_path)

    texts = texts.fillna("").astype(str).tolist()
    all_embeddings = []

    print(f"\nExtracting AI BERT embeddings for {split_name}: {len(texts)} rows")

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
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(emb.float().cpu().numpy())

        if start % 200 == 0:
            print(f"{split_name}: processed {start}/{len(texts)}")

    arr = np.vstack(all_embeddings).astype(np.float32)
    np.save(save_path, arr)

    print("Saved:", save_path, arr.shape)
    return arr


X_ai_train_bert = extract_ai_embeddings(ai_train["raw_email"], "ai_train")
X_ai_val_bert = extract_ai_embeddings(ai_val["raw_email"], "ai_val")
X_ai_test_bert = extract_ai_embeddings(ai_test["raw_email"], "ai_test")


# =========================
# Structural features
# =========================

print("\nPreparing structural features...")

for df_part in [ai_train, ai_val, ai_test]:
    struct_rows = df_part["raw_email"].apply(compute_structural_from_raw).tolist()
    struct_df = pd.DataFrame(struct_rows)
    for col in STRUCTURAL_FEATURES:
        df_part[col] = struct_df[col].values

# Fit new scaler on original train + AI train
orig_train_struct_raw = train_df[STRUCTURAL_FEATURES].copy()
ai_train_struct_raw = ai_train[STRUCTURAL_FEATURES].copy()

combined_train_struct_raw = pd.concat(
    [orig_train_struct_raw, ai_train_struct_raw],
    ignore_index=True
)

_, scaler = prepare_structural_matrix(combined_train_struct_raw, fit_scaler=True)

X_train_struct, _ = prepare_structural_matrix(train_df, scaler=scaler, fit_scaler=False)
X_val_struct, _ = prepare_structural_matrix(val_df, scaler=scaler, fit_scaler=False)
X_test_struct, _ = prepare_structural_matrix(test_df, scaler=scaler, fit_scaler=False)

X_ai_train_struct, _ = prepare_structural_matrix(ai_train, scaler=scaler, fit_scaler=False)
X_ai_val_struct, _ = prepare_structural_matrix(ai_val, scaler=scaler, fit_scaler=False)
X_ai_test_struct, _ = prepare_structural_matrix(ai_test, scaler=scaler, fit_scaler=False)

joblib.dump(scaler, os.path.join(NEW_OUTPUT_DIR, "hybrid_ai_adapted_structural_scaler.pkl"))


# =========================
# Combine features
# =========================

X_train_orig = np.hstack([X_train_bert, X_train_struct]).astype(np.float32)
X_val_orig = np.hstack([X_val_bert, X_val_struct]).astype(np.float32)
X_test_orig = np.hstack([X_test_bert, X_test_struct]).astype(np.float32)

X_ai_train = np.hstack([X_ai_train_bert, X_ai_train_struct]).astype(np.float32)
X_ai_val = np.hstack([X_ai_val_bert, X_ai_val_struct]).astype(np.float32)
X_ai_test = np.hstack([X_ai_test_bert, X_ai_test_struct]).astype(np.float32)

y_train_orig = train_df["label"].astype(int).values
y_val_orig = val_df["label"].astype(int).values
y_test_orig = test_df["label"].astype(int).values

y_ai_train = ai_train["label"].astype(int).values
y_ai_val = ai_val["label"].astype(int).values
y_ai_test = ai_test["label"].astype(int).values

# Repeat AI training samples so they are not drowned by original data
X_ai_train_repeated = np.repeat(X_ai_train, AI_REPEAT, axis=0)
y_ai_train_repeated = np.repeat(y_ai_train, AI_REPEAT, axis=0)

X_train_combined = np.vstack([X_train_orig, X_ai_train_repeated]).astype(np.float32)
y_train_combined = np.concatenate([y_train_orig, y_ai_train_repeated]).astype(int)

print("\nTraining feature shapes:")
print("Original train:", X_train_orig.shape)
print("AI train repeated:", X_ai_train_repeated.shape)
print("Combined train:", X_train_combined.shape)


# =========================
# Initialize from old hybrid model
# =========================

input_dim = X_train_combined.shape[1]
model = HybridMLP(input_dim=input_dim).to(device)

old_state = safe_load_state_dict(OLD_MLP_PATH)
model.load_state_dict(old_state)

print("\nLoaded old hybrid MLP weights for adaptation.")


# =========================
# Baseline before adaptation
# =========================

print("\nBaseline old hybrid performance before adaptation:")

_, _, _, _, y0_orig, p0_orig, _ = evaluate_model(model, X_test_orig, y_test_orig)
print_report("Original GP2 Test - Before Adaptation", y0_orig, p0_orig)

_, _, _, _, y0_ai, p0_ai, _ = evaluate_model(model, X_ai_test, y_ai_test)
print_report("AI Holdout Test - Before Adaptation", y0_ai, p0_ai)


# =========================
# Adapt training
# =========================

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train_combined, dtype=torch.float32),
        torch.tensor(y_train_combined, dtype=torch.long)
    ),
    batch_size=MLP_BATCH_SIZE,
    shuffle=True
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

best_score = -1
epochs_without_improvement = 0
best_model_path = os.path.join(NEW_OUTPUT_DIR, "hybrid_bert_mlp_ai_adapted.pt")

start_time = time.time()

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

    orig_val_acc, orig_val_precision, orig_val_recall, orig_val_f1, _, _, _ = evaluate_model(
        model, X_val_orig, y_val_orig
    )

    ai_val_acc, ai_val_precision, ai_val_recall, ai_val_f1, _, _, _ = evaluate_model(
        model, X_ai_val, y_ai_val
    )

    # We care about AI improvement, but still want original validation to remain strong.
    selection_score = ai_val_f1 + 0.50 * orig_val_f1

    print(
        f"Epoch {epoch:02d} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Orig Val F1: {orig_val_f1:.4f}, Recall: {orig_val_recall:.4f} | "
        f"AI Val F1: {ai_val_f1:.4f}, Recall: {ai_val_recall:.4f} | "
        f"Score: {selection_score:.4f}"
    )

    if selection_score > best_score:
        best_score = selection_score
        epochs_without_improvement = 0
        torch.save(model.state_dict(), best_model_path)
        print("Saved best adapted model.")
    else:
        epochs_without_improvement += 1
        print(f"No improvement: {epochs_without_improvement}/{PATIENCE}")

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered.")
        break

end_time = time.time()

print("\nAdaptation training time:")
print(f"{end_time - start_time:.2f} seconds")


# =========================
# Final evaluation
# =========================

model.load_state_dict(safe_load_state_dict(best_model_path))

orig_acc, orig_precision, orig_recall, orig_f1, y_orig_true, y_orig_pred, y_orig_probs = evaluate_model(
    model, X_test_orig, y_test_orig
)

ai_acc, ai_precision, ai_recall, ai_f1, y_ai_true, y_ai_pred, y_ai_probs = evaluate_model(
    model, X_ai_test, y_ai_test
)

print_report("Original GP2 Test - After AI Adaptation", y_orig_true, y_orig_pred)
print_report("AI Holdout Test - After AI Adaptation", y_ai_true, y_ai_pred)


# =========================
# Save predictions and summary
# =========================

orig_test_results = test_df.copy()
orig_test_results["true_label"] = y_orig_true
orig_test_results["predicted_label"] = y_orig_pred
orig_test_results["prob_legitimate"] = y_orig_probs[:, 0]
orig_test_results["prob_phishing"] = y_orig_probs[:, 1]
orig_test_results["prediction_type"] = "correct"

orig_test_results.loc[
    (orig_test_results["true_label"] == 0) & (orig_test_results["predicted_label"] == 1),
    "prediction_type"
] = "false_positive"

orig_test_results.loc[
    (orig_test_results["true_label"] == 1) & (orig_test_results["predicted_label"] == 0),
    "prediction_type"
] = "false_negative"

orig_test_results.to_csv(
    os.path.join(NEW_OUTPUT_DIR, "adapted_original_test_predictions.csv"),
    index=False
)

ai_test_results = ai_test.copy()
ai_test_results["true_label"] = y_ai_true
ai_test_results["predicted_label"] = y_ai_pred
ai_test_results["prob_legitimate"] = y_ai_probs[:, 0]
ai_test_results["prob_phishing"] = y_ai_probs[:, 1]
ai_test_results["prediction_type"] = "correct"

ai_test_results.loc[
    (ai_test_results["true_label"] == 0) & (ai_test_results["predicted_label"] == 1),
    "prediction_type"
] = "false_positive"

ai_test_results.loc[
    (ai_test_results["true_label"] == 1) & (ai_test_results["predicted_label"] == 0),
    "prediction_type"
] = "false_negative"

ai_test_results.to_csv(
    os.path.join(NEW_OUTPUT_DIR, "adapted_ai_holdout_test_predictions.csv"),
    index=False
)

summary = {
    "experiment": "AI-generated email adaptation",
    "base_model": "hybrid_bert_embeddings_structural_mlp",
    "adaptation_method": "Continue training MLP classifier using original training data plus repeated AI-generated training split",
    "ai_repeat": AI_REPEAT,
    "original_test": {
        "accuracy": float(orig_acc),
        "precision": float(orig_precision),
        "recall": float(orig_recall),
        "f1": float(orig_f1),
        "confusion_matrix": confusion_matrix(y_orig_true, y_orig_pred).tolist()
    },
    "ai_holdout_test": {
        "accuracy": float(ai_acc),
        "precision": float(ai_precision),
        "recall": float(ai_recall),
        "f1": float(ai_f1),
        "confusion_matrix": confusion_matrix(y_ai_true, y_ai_pred).tolist()
    }
}

with open(os.path.join(NEW_OUTPUT_DIR, "ai_adaptation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("\nSaved adapted model and outputs to:")
print(NEW_OUTPUT_DIR)