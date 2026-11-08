# Reported evaluation results

These are **historical research results** from the group graduation report and provided aggregate result files. They have **not been reproduced** from checkpoints in this repository.

## Internal evaluation (31,063 test messages)

| Model | Accuracy | False positives | False negatives |
|---|---:|---:|---:|
| SVM | 98.60% | 222 | 199 |
| BERT | 99.48% | 56 | 105 |
| RoBERTa | 99.43% | 40 | 138 |
| Hybrid BERT + structural features | 99.48% | 63 | 98 |

The hybrid model was selected to prioritize minimizing missed phishing messages. RoBERTa had fewer false positives on the internal test.

## External blind test (322,601 messages)

| Actual / predicted | Legitimate | Phishing |
|---|---:|---:|
| Legitimate | 135,829 | 32,625 |
| Phishing | 1,441 | 152,706 |

**Accuracy:** 288,535 / 322,601 ≈ 89.44%. High false positive counts raise important domain shift, data provenance and labeling concerns and should not be omitted from portfolio discussions.

## Human vs AI-generated messages (4,000 total)

| Subset | Total | Correct | False positives | False negatives |
|---|---:|---:|---:|---:|
| Human-written | 2,000 | 2,000 | 0 | 0 |
| AI-generated | 2,000 | 1,351 | 522 | 127 |
| All | 4,000 | 3,351 | 522 | 127 |

Overall accuracy ≈ **83.78%**; accuracy on AI-generated subset was **67.55%**. These results do not establish performance on production email.
