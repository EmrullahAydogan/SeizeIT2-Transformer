# SeizeIT2: Transformer-Based Seizure Detection

🧠 **Unsupervised anomaly detection for epileptic seizure identification using Transformer-Autoencoder architecture**

## 📋 Overview

This is a pilot study implementing a Transformer-based autoencoder for real-time epilepsy seizure detection using multi-modal physiological signals (EEG, ECG, EMG, Movement).

**Key Features:**
- ✅ Unsupervised learning approach (learns from normal data only)
- ✅ Multi-modal signal fusion (16-channel EEG + ECG + EMG + MOV)
- ✅ Transformer architecture with self-attention mechanism
- ✅ Academic-grade evaluation (ROC, AUC, sensitivity, specificity)
- ✅ Per-patient performance analysis

## 🎯 Dataset

**Source:** SeizeIT2 Dataset (125 patients with focal epilepsy)

**Selected Patients (n=3):**
| Patient | Duration (h) | Seizures | Vigilance | Seizure Type |
|---------|-------------|----------|-----------|--------------|
| sub-015 | 20.9 | 9 | Mixed | Tonic |
| sub-022 | 21.5 | 7 | Mixed | Automatisms |
| sub-103 | 18.4 | 15 | Mixed | Hyperkinetic |
| **Total** | **60.8** | **31** | - | - |

Selection criteria:
- Duration > 18 hours (circadian rhythm coverage)
- Mixed vigilance states (both sleep and wake seizures)
- Multiple seizure types (diversity)
- Multi-modal sensor availability

## 🏗️ Architecture

**Model:** Transformer-Autoencoder
- Input: 16 channels × 1000 timesteps (4 seconds @ 250 Hz)
- Embedding: 1D-CNN (64 filters, kernel=5)
- Attention: 4 heads, 64 dimensions
- Decoder: 1D-CNN reconstruction
- Loss: Mean Squared Error (MSE)

**Training:**
- Optimizer: Adam (lr=1e-4)
- Epochs: 5 (baseline, will be extended to 50-100)
- Batch size: 16
- Hardware: RTX 4070 8GB

## 📊 Results

*Results will be updated after running `evaluate_per_patient.m`*

**Preliminary Performance (per-patient):**
- AUC: TBD
- Sensitivity: TBD
- Specificity: TBD
- F1-Score: TBD

## 🚀 Usage

### Prerequisites
```matlab
% MATLAB R2020b+ with:
% - Deep Learning Toolbox
% - Signal Processing Toolbox
% - Statistics and Machine Learning Toolbox
```

### Quick Start

```matlab
% 1. Data Preprocessing (requires SeizeIT2 dataset)
run('MatlabProject/process_final_dataset.m');

% 2. Create Training Data
run('MatlabProject/create_model_data.m');

% 3. Train Model
run('MatlabProject/train_transformer_autoencoder.m');

% 4. Evaluate (Academic Analysis)
run('MatlabProject/evaluate_per_patient.m');

% 5. Visualize Explainability
run('MatlabProject/visualize_explainability.m');
```

## 📁 Project Structure

```
SeizeIT2/
├── MatlabProject/
│   ├── train_transformer_autoencoder.m  # Model training
│   ├── evaluate_per_patient.m           # Academic evaluation ⭐ NEW
│   ├── process_final_dataset.m          # Data preprocessing
│   ├── create_model_data.m              # Windowing & splitting
│   ├── visualize_explainability.m       # Attention visualization
│   ├── compare_normal_vs_seizure.m      # Baseline comparison
│   ├── generate_academic_matrix.m       # Patient selection analysis
│   ├── ModelData/
│   │   ├── Trained_Transformer_Final.mat
│   │   ├── Train/*.mat                  # Training windows
│   │   └── Test/*.mat                   # Test windows
│   ├── ProcessedData/*.mat              # Preprocessed signals
│   └── Results/                         # Evaluation outputs
│       ├── ROC_Curves_PerPatient.png
│       ├── Confusion_Matrices.png
│       └── PerPatient_Performance.csv
└── dataset/                             # SeizeIT2 raw data (not included)
```

## ⚠️ Limitations

**Hardware Constraints:**
- GPU Memory: 8GB limits cohort size to 3 patients
- Future work: Cloud-based scaling (AWS/GCP)

**Statistical:**
- Small sample size (n=3) → This is a **pilot study**
- No cross-validation yet (planned: LOPO-CV)

**Clinical:**
- No real-time latency analysis
- No prospective validation
- Requires clinician review before deployment

## 🔮 Future Work

**Short-term:**
- [ ] Leave-One-Patient-Out Cross-Validation
- [ ] Extended training (50-100 epochs)
- [ ] Threshold optimization per-patient
- [ ] Ablation study (window sizes, modalities)

**Long-term:**
- [ ] Scale to 10-15 patients (cloud GPU)
- [ ] Real-time inference pipeline
- [ ] Attention weight visualization
- [ ] Clinical validation study

## 📚 Citation

*Paper in preparation*

```bibtex
@misc{seizeit2_transformer2025,
  title={Transformer-Based Anomaly Detection for Epileptic Seizure Identification: A Pilot Study},
  author={[Your Name]},
  year={2025},
  note={Code available at: https://github.com/[username]/SeizeIT2-Transformer}
}
```

## 📄 License

MIT License (or specify your preference)

## 🙏 Acknowledgments

- SeizeIT2 Dataset: [Original dataset authors]
- Hardware: RTX 4070 8GB

---

**Status:** 🟡 In Development (Pilot Study Phase)

**Last Updated:** January 2025
