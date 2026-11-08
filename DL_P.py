import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BERT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_full_model")

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)

model.to(device)
model.eval()

text = """
Subject: Account verification required
Body: Please verify your account immediately to avoid suspension.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

pred = torch.argmax(probs, dim=1).item()

print("Prediction:", "phishing" if pred == 1 else "legitimate")
print("Legitimate probability:", probs[0][0].item())
print("Phishing probability:", probs[0][1].item())