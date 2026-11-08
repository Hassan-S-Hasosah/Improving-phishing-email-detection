# Security and release review

This repository is **private** while the code and rights are reviewed.

- Uploaded source files received an initial pattern-based credential scan; that cannot guarantee absence of secrets.
- Never commit API keys, access tokens, passwords, or private keys.
- Rotate keys if they were previously exposed; simply removing them from Git history is not enough.
- Training email datasets, per-row predictions, pretrained weights, and embedding caches are excluded from this repository.
- Original user-specific Windows paths in Python scripts have been changed to paths relative to the source files.
- Flask CORS and an unauthenticated local endpoint are for demonstration only. Do not expose this system publicly without authentication, rate limiting, privacy safeguards, and security review.
- The frontend was written by teammates; this repository does not include their frontend source.
