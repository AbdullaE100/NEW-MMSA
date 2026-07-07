# Contributing

Thanks for taking a look. This is a research prototype, so the most valuable
contributions are the ones that make it more honest and more correct — a learned
fusion head with reported numbers is worth more than another UI tweak.

## Setup

```bash
git lfs install
git clone https://github.com/AbdullaE100/NEW-MMSA.git
cd NEW-MMSA
git lfs pull
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest ruff
```

## Before you open a PR

```bash
pytest tests/ -v
ruff check .
```

Both run in CI on every push and pull request.

## Guidelines

- **Keep the unit tests dependency-light.** Tests in `tests/` must not import
  TensorFlow, torch, or DeepFace, and must not need the model weights. Put
  testable logic in a pure module (like `src/audio_features.py`) and test that.
- **No label leakage, ever.** Predictions must depend only on clip content —
  never on filenames, directory names, or accompanying label files. Ground-truth
  labels may be read only to *score* predictions, never to influence them.
- **Inference must be deterministic.** No `np.random` / `random` in a code path
  that produces a prediction.
- **Name things for what they are.** If it isn't SHAP, don't call it SHAP in
  anything a user reads.
- **Put tunable constants in `src/config.py`** with a comment on what the value
  means, rather than as inline literals.

## Good first issues

- Add an `eval/` harness that reports macro-F1 on a RAVDESS held-out split.
- Replace the hand-tuned fusion weights with a small trained logistic head.
- Remove a per-modality "boost" heuristic and show the effect on evaluation.
