# Layer-wise Cross-Lingual Depression Detection from Speech: A HuBERT-Based Study on English and Mandarin

This repository contains the code and experiment pipeline for the bachelor's thesis project **“Cross-Lingual Speech-Based Depression Detection using HuBERT”** at the University of Groningen. The project investigates whether depression-related speech cues are transferable across languages using HuBERT representations.

## 📘 Overview

Recent self-supervised speech models like HuBERT have demonstrated strong performance on emotion recognition tasks. This project explores whether **mid-layer HuBERT embeddings** encode language-independent acoustic cues relevant to **clinical depression**, and evaluates this via:

- **Monolingual classification** (EN, ZH, MIX)
- **Cross-lingual transfer** (e.g., EN→ZH)
- **Layer-wise comparison** (Layers 6–9)
- **Simple baseline classifier** (Logistic Regression)

## 🗂️ Project Structure
```
HuBERT_Depression_Detection/
├── code/
│   ├── classification/
│   │   ├── run_cmdc_logistic_classifier.py
│   │   ├── run_edaic_logistic_classifier.py
│   │   └── run_mix_logistic_classifier.py
│   ├── feature_extraction/
│   │   ├── extract_cmdc_layer.py
│   │   ├── extract_edaic_layer.py
│   │   └── extract_mix_layer.py
│   └── preprocessing/
│       ├── balance_utterance_table.py
│       ├── build_mixed_metadata.py
│       ├── create_cmdc_balanced_table.py
│       ├── create_cmdc_utterance_table.py
│       ├── cut_edaic_utterances.py
│       ├── segment_cmdc_sliding.py
│       ├── segment_edaic_sliding.py
│       └── split_metadata.py
├── features/       
├── log/              
├── README.md
└── .gitignore
```


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
| EN Layer 6 | 0.74     | 0.803     | 0.883    |
| ZH Layer 6 | 0.954     | 0.96     | 0.993    |
| MIX Layer 6| 0.801     | 0.833     | 0.919    |

## 📁 Reproducibility

- Each step is modular and resume-friendly.
- Scripts support argument-based configuration for easy extension.
- Use `nohup` for GPU-based training/feature extraction in background.

## 📄 Citation

If you use or reference this project, please cite:

> Hang Chen (2025). *Layer-wise Cross-Lingual Depression Detection from Speech: A HuBERT-Based Study on English and Mandarin*. Master's Thesis, University of Groningen.

## 🧠 Acknowledgments

- Datasets: DAIC-WOZ, CMDC
- Libraries: HuggingFace Transformers, Torchaudio, scikit-learn, Pandas

---

📬 For questions, contact: **hchen90408@gmail.com**
