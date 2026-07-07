# Architecture

A tour of how a video clip becomes a sentiment score, and why the code is laid
out the way it is.

## Request flow

1. A clip arrives through the Gradio UI (`mmsa_interface_with_shap.py`) or the
   batch CLI (`run_mmsa.py batch`).
2. `MultimodalSentimentGradio` (`src/mmsa_gradio_interface.py`) orchestrates the
   three analyzers:
   - **Visual** — `DeepFaceEmotionDetector` samples frames and runs DeepFace to
     get facial-emotion probabilities, mapped to a sentiment score.
   - **Audio** — `AudioSentimentAnalyzer` extracts the track, computes MFCCs
     over a fixed-length window, and runs the CNN.
   - **Text** — the audio is transcribed and `TextSentimentAnalyzer` runs
     RoBERTa over the transcript.
3. The three scores are combined by weighted late fusion. Missing modalities are
   dropped and the remaining weights renormalised.
4. `MultimodalSentimentWithSHAP` wraps the base interface and adds the
   per-modality **feature-contribution charts** (weighted bar charts, not SHAP).

## Layers

| Layer | Files | Responsibility |
|-------|-------|----------------|
| Entry points | `mmsa_interface_with_shap.py`, `app.py`, `run_mmsa.py` | Wire up and launch; no analysis logic |
| Orchestration | `src/mmsa_gradio_interface.py` | Call analyzers, fuse scores |
| Analyzers | `src/mmsa_audio_sentiment.py`, `src/deepface_emotion_detector.py`, `src/mmsa_text_sentiment.py` | One modality each |
| Pure helpers | `src/audio_features.py`, `src/config.py` | Testable logic, tunable constants |
| Batch | `src/batch_mmsa_processor.py` | Run analyzers over a directory |

## Design decisions

**Why late fusion instead of a joint model?** The three modalities have very
different feature shapes and pretrained backbones. Late fusion lets each use the
best available off-the-shelf model and keeps the components independently
testable and swappable. The cost is that the combination rule is not learned —
the current hand-tuned weights are the weakest link (see MODEL_CARD.md).

**Why is length-normalisation its own module?** The audio CNN needs a
fixed-size input. That framing logic used to live inline in the feature
extractor and used a random offset, which made inference non-deterministic. It's
now a pure function in `src/audio_features.py` with no heavy dependencies, so it
can be unit-tested in isolation and can't silently regress.

**Why does the wrapper class keep the "SHAP" name?** Renaming the class and
methods would ripple through several call sites for no functional gain. Instead
every user-facing label describes the charts accurately as "feature
contributions," and a module-level note explains the historical naming. The
honest label is what the user sees; the internal identifier is legacy.

## Known rough edges

- The base interface and the SHAP wrapper overlap in responsibility; a cleaner
  design would collapse them into one interface with an optional charting hook.
- Sentiment thresholds and the heuristic "boosts" are still partly inline in the
  analyzers rather than fully centralised in `config.py`.
- Result HTML is assembled as strings in Python; moving to Gradio components
  would be more robust.

These are tracked as cleanup work, not hidden.
