# Data and model artifact requirements

The code in this repository is **not immediately runnable**: dataset CSVs, saved transformers, model weights and scaler artifacts are intentionally not included.

## Expected local files

```text
bert_full_model/                       # trained BERT checkpoint and tokenizer
roberta_full_model/                    # optional saved RoBERTa model
hybrid_full_outputs/
  hybrid_bert_mlp_full.pt              # trained hybrid classifier weights
  hybrid_full_structural_scaler.pkl    # fitted structural feature scaler
train.csv
validation.csv
test.csv
df.csv                                 # external test dataset
AI_legit.csv
AI_phishing.csv
H_legit.csv
H_phishing.csv
```

Scripts use the directory containing the Python source as their project folder. Dataset paths, pretrained model artifacts, and training hyperparameters should be checked before running large jobs.

## Local API (with artifacts supplied)

```bash
python -m venv .venv
pip install -r requirements.txt
python api.py
```

Call `POST http://127.0.0.1:5000/predict` with JSON containing `email_text` and optional `mode` (`balanced` or `high_security`). The original team-developed frontend source is not included. This academic prototype is not production-ready.

Dataset licensing, third-party permissions, record-level privacy, and output provenance must be reviewed independently before sharing data or model weights.
