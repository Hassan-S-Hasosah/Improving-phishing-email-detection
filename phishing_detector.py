import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Hybrid BERT + Structural Features Phishing Detector
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BERT_MODEL_DIR = os.path.join(BASE_DIR, "bert_full_model")
HYBRID_OUTPUT_DIR = os.path.join(BASE_DIR, "hybrid_full_outputs")

HYBRID_MODEL_PATH = os.path.join(HYBRID_OUTPUT_DIR, "hybrid_bert_mlp_full.pt")
SCALER_PATH = os.path.join(HYBRID_OUTPUT_DIR, "hybrid_full_structural_scaler.pkl")

MAX_LENGTH = 256

THRESHOLDS = {
    "balanced": 0.50,
    "high_security": 0.24
}

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


class PhishingDetector:
    def __init__(
        self,
        bert_model_dir=BERT_MODEL_DIR,
        hybrid_model_path=HYBRID_MODEL_PATH,
        scaler_path=SCALER_PATH,
        max_length=MAX_LENGTH
    ):

        self.max_length = max_length
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print("Loading phishing detector...")
        print("Device:", self.device)
        
        if self.device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_dir)

        bert_classifier = AutoModelForSequenceClassification.from_pretrained(bert_model_dir)
        self.bert_model = bert_classifier.bert
        self.bert_model.to(self.device)
        self.bert_model.eval()

        self.scaler = joblib.load(scaler_path)

        state_dict = self._safe_load_state_dict(hybrid_model_path)

        input_dim = state_dict["net.0.weight"].shape[1]

        self.hybrid_model = HybridMLP(input_dim=input_dim)
        self.hybrid_model.load_state_dict(state_dict)
        self.hybrid_model.to(self.device)
        self.hybrid_model.eval()

        print("Detector loaded successfully.")

    def _safe_load_state_dict(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)

    @staticmethod
    def extract_urls(text):
        if not text:
            return []

        url_pattern = r"(https?://\S+|www\.\S+)"
        return re.findall(url_pattern, text)

    @staticmethod
    def build_text(subject, body):
        subject = subject or ""
        body = body or ""

        return f"Subject: {subject.strip()}\nBody: {body.strip()}".strip()

    def parse_raw_email(self, raw_email):
        raw_email = raw_email or ""

        subject = ""
        sender = ""
        receiver = ""

        lines = raw_email.splitlines()
        body_lines = []

        for line in lines:
            clean_line = line.strip()
            lower = clean_line.lower()

            if lower.startswith("subject:") and subject == "":
                subject = clean_line.split(":", 1)[1].strip()

            elif lower.startswith("from:") and sender == "":
                sender = clean_line.split(":", 1)[1].strip()

            elif lower.startswith("to:") and receiver == "":
                receiver = clean_line.split(":", 1)[1].strip()

            else:
                body_lines.append(line)

        body = "\n".join(body_lines).strip()

        if body == "":
            body = raw_email.strip()

        return {
            "subject": subject,
            "body": body,
            "sender": sender,
            "receiver": receiver
        }

    def compute_structural_features(
        self,
        subject,
        body,
        sender="",
        receiver="",
        urls=None
    ):
        subject = subject or ""
        body = body or ""
        sender = sender or ""
        receiver = receiver or ""

        text = self.build_text(subject, body)

        if urls is None:
            detected_urls = self.extract_urls(subject + " " + body)
        elif isinstance(urls, list):
            detected_urls = urls
        else:
            detected_urls = self.extract_urls(str(urls))

        url_count = len(detected_urls)

        raw_features = {
            "url_present": 1 if url_count > 0 else 0,
            "url_count": url_count,
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

        features_df = pd.DataFrame([raw_features], columns=STRUCTURAL_FEATURES)

        for col in STRUCTURAL_FEATURES:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce").fillna(0)

        features_df[NUMERIC_FEATURES] = np.log1p(features_df[NUMERIC_FEATURES])
        features_df[NUMERIC_FEATURES] = self.scaler.transform(features_df[NUMERIC_FEATURES])
        features_df[BINARY_FEATURES] = features_df[BINARY_FEATURES].astype(float)

        struct_array = features_df.values.astype(np.float32)

        return struct_array, raw_features

    def extract_bert_embedding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.bert_model(**inputs)
            else:
                outputs = self.bert_model(**inputs)

            if outputs.pooler_output is not None:
                embedding = outputs.pooler_output
            else:
                embedding = outputs.last_hidden_state[:, 0, :]

        return embedding.float().cpu().numpy().astype(np.float32)

    def predict(
        self,
        subject,
        body,
        sender="",
        receiver="",
        urls=None,
        mode="balanced"
    ):
        if mode not in THRESHOLDS:
            mode = "balanced"

        threshold = THRESHOLDS[mode]

        text = self.build_text(subject, body)

        bert_embedding = self.extract_bert_embedding(text)

        struct_features, raw_structural_features = self.compute_structural_features(
            subject=subject,
            body=body,
            sender=sender,
            receiver=receiver,
            urls=urls
        )

        hybrid_input = np.hstack([bert_embedding, struct_features]).astype(np.float32)
        hybrid_tensor = torch.tensor(hybrid_input, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.hybrid_model(hybrid_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        prob_legitimate = float(probs[0])
        prob_phishing = float(probs[1])

        predicted_label = 1 if prob_phishing >= threshold else 0
        prediction = "phishing" if predicted_label == 1 else "legitimate"

        risk_level = self.get_risk_level(prob_phishing)

        return {
            "prediction": prediction,
            "predicted_label": predicted_label,
            "mode": mode,
            "threshold": threshold,
            "prob_legitimate": prob_legitimate,
            "prob_phishing": prob_phishing,
            "risk_level": risk_level,
            "structural_features": raw_structural_features
        }

    def predict_raw_email(self, raw_email, mode="balanced"):
        parsed = self.parse_raw_email(raw_email)

        return self.predict(
            subject=parsed["subject"],
            body=parsed["body"],
            sender=parsed["sender"],
            receiver=parsed["receiver"],
            mode=mode
        )

    @staticmethod
    def get_risk_level(prob_phishing):
        if prob_phishing >= 0.80:
            return "high"
        elif prob_phishing >= 0.50:
            return "medium"
        elif prob_phishing >= 0.24:
            return "low-to-medium"
        else:
            return "low"


if __name__ == "__main__":
    detector = PhishingDetector()

    sample_email = """
From: security@bank-alerts-secure.com
To: user@email.com
Subject: URGENT: Your account has been compromised!

Dear Customer,

We detected suspicious activity on your account.
Please verify your account immediately by visiting:
https://secure-bank-login-example.com

Failure to verify may result in account suspension.
"""

    result_balanced = detector.predict_raw_email(
        raw_email=sample_email,
        mode="balanced"
    )

    result_high_security = detector.predict_raw_email(
        raw_email=sample_email,
        mode="high_security"
    )

    print("\nBalanced mode result:")
    print(result_balanced)

    print("\nHigh security mode result:")
    print(result_high_security)