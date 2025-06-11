# Cross-Lingual Speech-Based Depression Detection using HuBERT

This repository contains the code and experiment pipeline for the bachelor's thesis project **“Cross-Lingual Speech-Based Depression Detection using HuBERT”** at the University of Groningen. The project investigates whether depression-related speech cues are transferable across languages using HuBERT representations.

## 📘 Overview

Recent self-supervised speech models like HuBERT have demonstrated strong performance on emotion recognition tasks. This project explores whether **mid-layer HuBERT embeddings** encode language-independent acoustic cues relevant to **clinical depression**, and evaluates this via:

- **Monolingual classification** (EN, ZH, MIX)
- **Cross-lingual transfer** (e.g., EN→ZH)
- **Layer-wise comparison** (Layers 6–9)
- **Simple baseline classifier** (Logistic Regression)

## 🗂️ Project Structure
HuBERT/
├── datasets/                             # Segmented .wav files from CMDC & E-DAIC
│   ├── cmdc_segments/
│   └── edaic_segments/
├── features_cmdc/                        # CMDC HuBERT features (e.g., X_train.npy, y_train.npy)
├── features_edaic/                       # E-DAIC HuBERT features
├── features_mix/                         # Mixed-language HuBERT features
├── extract_cmdc_layer.py                # CMDC feature extraction script (layer parameterized)
├── extract_edaic_layer.py               # E-DAIC feature extraction script (layer parameterized)
├── extract_mix_layer.py                 # MIX feature extraction script (layer parameterized)
├── run_cmdc_logistic_classifier.py      # Monolingual logistic regression for CMDC
├── run_edaic_logistic_classifier.py     # Monolingual logistic regression for E-DAIC
├── run_mix_logistic_classifier.py       # Monolingual logistic regression for MIX
├── run_crosslingual_logistic.py         # Cross-lingual evaluation (EN→ZH, ZH→EN, MIX→EN/ZH)
├── segment_cmdc_sliding.py              # CMDC preprocessing with 3s sliding window + downsampling
├── segment_edaic_sliding.py             # E-DAIC preprocessing with 3s sliding window + downsampling
├── build_mixed_metadata.py              # Sampled metadata for balanced MIX dataset
├── utterance_table_cmdc_segmented_split.csv
├── utterance_table_edaic_segmented_split.csv
├── utterance_table_mix_segmented_split.csv

## 🔍 Datasets

- **E-DAIC**: Extended DAIC-WOZ Corpus for English depression speech, PHQ-8 based.
- **CMDC**: Chinese Multimodal Depression Corpus with binary PHQ-9-derived labels.
- All audio was segmented using **3-second sliding windows with 50% overlap**.

## 🛠️ Pipeline

1. **Preprocessing**
   - Downsampling to 16kHz
   - Segmentation using a 3s sliding window
   - Balanced sampling across classes

2. **HuBERT Feature Extraction**
   - Extracted using HuggingFace `facebook/hubert-base-ls960`
   - Layers 6–9, 768-dim mean pooled embeddings
   - Implemented in `extract_*_layer.py`

3. **Classification**
   - Baseline: `LogisticRegression` from scikit-learn
   - Class weighting for imbalance (`class_weight='balanced'`)
   - Evaluation: F1, Accuracy, ROC-AUC

4. **Cross-Lingual Transfer**
   - Train on EN / ZH / MIX → Test on other languages
   - Evaluated via `run_crosslingual_logistic.py`

## 📊 Results

- Monolingual F1 scores peak at Layer 6
- MIX-trained models show improved generalization
- EN→ZH and ZH→EN both show non-trivial performance
- Logistic regression enables controlled evaluation of HuBERT layer quality

| Model      | F1 (avg) | Accuracy | ROC-AUC |
|------------|----------|----------|---------|
| EN Layer 6 | 0.54     | 0.76     | 0.79    |
| ZH Layer 6 | 0.66     | 0.72     | 0.74    |
| MIX Layer 6| 0.67     | 0.76     | 0.78    |

## 📁 Reproducibility

- Each step is modular and resume-friendly.
- Scripts support argument-based configuration for easy extension.
- Use `nohup` for GPU-based training/feature extraction in background.

## 📄 Citation

If you use or reference this project, please cite:

> Hang Chen (2025). *Cross-Lingual Speech-Based Depression Detection using HuBERT*. Master's Thesis, University of Groningen.

## 🧠 Acknowledgments

- Datasets: DAIC-WOZ, CMDC
- Libraries: HuggingFace Transformers, Torchaudio, scikit-learn, Pandas

---

📬 For questions, contact: **h.chen.49@student.rug.nl**
