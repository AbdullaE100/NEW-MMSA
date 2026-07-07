# Model Card — Multimodal Sentiment Analysis

This card describes the composite system, not a single trained model. The system
stitches together three off-the-shelf/pretrained components with a hand-tuned
fusion rule.

## Intended use

Exploratory analysis of sentiment in short (roughly 5–30 second) video clips of a
single speaking person. Suitable for demos, teaching multimodal fusion, and
research prototyping.

**Not** suitable for: automated decisions about people, moderation, hiring,
clinical or affective assessment, or any setting where a wrong sentiment label
carries real consequences. It has no validated accuracy and known biases.

## Components

| Component | Source | Training data | Notes |
|-----------|--------|---------------|-------|
| Visual | DeepFace (pretrained emotion model) | FER-2013-style data | Emotion, not sentiment; mapped to sentiment downstream |
| Audio | 2-D CNN in this repo | RAVDESS | Acted emotional speech, 24 actors, studio conditions |
| Text | `cardiffnlp/twitter-roberta-base-sentiment-latest` | ~124M tweets | Trained on social-media English |

## Fusion

Late fusion by weighted average of the three per-modality sentiment scores.
Default weights (`src/config.py`): **visual 0.45, audio 0.45, text 0.10**,
renormalised over whichever modalities are present.

These weights were set by hand to look reasonable on sample clips. They were
**not** fit to labelled data and carry no statistical guarantee. This is the
single biggest caveat in the system.

### Known heuristic biases

The per-modality code contains hand-written adjustments that push certain
predictions around. They were added to fix under-detection on specific demo
clips and are **not** principled:

- Audio: positive "boosts" for `happy` / `surprised` / `calm` emotions.
- Visual: multiplicative boosts on `happy` / `surprise`.
- Text: a small confidence boost for non-neutral predictions.

These are documented rather than hidden. Removing them in favour of a learned
fusion head is on the roadmap.

## What was removed for integrity

Two behaviours were removed from earlier versions because they inflated apparent
quality:

1. **Filename label leakage.** The text path parsed emotion keywords out of the
   video filename (RAVDESS filenames encode the ground-truth label) and used
   them to override the model. This made accuracy look better than the model
   actually is. Removed — predictions now depend only on content.
2. **Random inference-time cropping.** Audio feature extraction used a random
   crop/pad offset, so the same clip produced different predictions on repeat
   runs. Replaced with a deterministic centered crop/pad
   (`src/audio_features.py`, unit-tested).

## Evaluation

None shipped yet. There is no held-out evaluation set or reported metric in this
repository. Any accuracy figure quoted elsewhere should be treated as
unverified until an `eval/` harness lands (see roadmap).

## Ethical considerations

- Facial- and voice-emotion recognition is contested; expressed emotion is not
  the same as felt emotion, and models carry demographic and cultural biases
  from their training data.
- RAVDESS is North American English acted speech; performance on other
  languages, accents, and natural (non-acted) speech is expected to be poor.
- Do not deploy this to assess real people.
