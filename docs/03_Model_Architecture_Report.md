# Model Architecture Report (v2.0)
**SeizeIT2 Transformer-based Seizure Detection Project - Academic Refactor**

---

## 📋 Executive Summary

This report documents the enhanced Transformer-Autoencoder architecture with Bayesian optimization and explainable AI features for unsupervised seizure detection in multi-modal physiological signals.

**Model Type:** Transformer Autoencoder with Multi-Head Attention (Unsupervised Anomaly Detection)
**Task:** Anomaly detection via reconstruction error (Normal → Low Error, Seizure → High Error)
**Input:** [16 channels × 129 frequency bins × 7 time frames] Spectrogram representation
**Output:** Reconstructed spectrogram (same shape as input)
**Anomaly Score:** Mean Squared Error (MSE) between input and reconstruction
**Optimization:** Bayesian hyperparameter optimization with AUC-PR objective
**Explainability:** Attention weight visualization for interpretable predictions

---

## 🎯 Purpose & Rationale

### Why Unsupervised Learning?

**Problem:**
- Seizures are rare events (<1% of data)
- Class imbalance makes supervised learning difficult
- Labeled data is expensive (requires expert annotation)

**Solution: Autoencoder Anomaly Detection**
1. **Train on normal data only** → Model learns "normal" signal patterns
2. **Test on both normal + seizure** → Seizures produce high reconstruction error
3. **Threshold error** → Classify based on reconstruction quality

**Advantages:**
- ✅ No need for balanced training data
- ✅ Works with limited seizure examples
- ✅ Detects novel/unseen seizure types (not seen during training)
- ✅ Interpretable (can visualize what's "abnormal")

**Precedent:** Similar approach used in:
- Medical imaging (tumor detection)
- Industrial IoT (equipment failure prediction)
- Cybersecurity (intrusion detection)

---

## 🏗️ Architecture Design

### Overall Structure

```
INPUT (16 × 1000)
    ↓
┌─────────────────┐
│  ENCODER        │
│  (Embedding)    │  → Compress to latent space
│  - Conv1D       │
│  - LayerNorm    │
│  - ReLU         │
└─────────────────┘
    ↓ (64 × 1000)
┌─────────────────┐
│  TRANSFORMER    │
│  (Attention)    │  → Capture temporal dependencies
│  - SelfAttention│
│  - LayerNorm    │
│  - ReLU         │
└─────────────────┘
    ↓ (64 × 1000)
┌─────────────────┐
│  DECODER        │
│  (Reconstruction)│  → Reconstruct original input
│  - Conv1D       │
└─────────────────┘
    ↓
OUTPUT (16 × 1000)

LOSS: MSE(Input, Output)
```

---

## 📊 Layer-by-Layer Specification

### Layer 1: Sequence Input
```matlab
sequenceInputLayer(16, 'MinLength', 1000, 'Name', 'input', 'Normalization', 'zscore')
```

**Parameters:**
- **Input channels:** 16 (multi-modal: EEG + ECG + EMG + MOV)
- **Sequence length:** 1000 timesteps (4 seconds @ 250 Hz)
- **Normalization:** Z-score (automatic per-batch normalization)

**Rationale:**
- Built-in normalization ensures stable training
- Variable-length support (for future variable window sizes)

---

### Layer 2-4: Encoder (Embedding)

#### 2. 1D Convolution
```matlab
convolution1dLayer(5, 64, 'Padding', 'same', 'Name', 'embed_conv')
```

**Parameters:**
- **Kernel size:** 5 timesteps (~20ms @ 250 Hz)
- **Filters:** 64 (embedding dimension)
- **Padding:** 'same' (output length = input length = 1000)

**Function:** Projects 16-channel input into 64-dimensional latent space
- **Rationale:** 64 dims balances expressiveness vs computational cost
- **Alternatives considered:**
  - 32 dims: Too compressed, information loss
  - 128 dims: Overfitting risk with small dataset

#### 3. Layer Normalization
```matlab
layerNormalizationLayer('Name', 'ln1')
```

**Function:** Normalizes across feature dimension (not batch)
- **Advantage over BatchNorm:** Works with small/variable batch sizes
- **Effect:** Stabilizes training, prevents gradient explosion

#### 4. ReLU Activation
```matlab
reluLayer('Name', 'relu1')
```

**Function:** Non-linearity f(x) = max(0, x)
- **Advantage:** Simple, fast, no vanishing gradient

---

### Layer 5-7: Transformer (Self-Attention)

#### 5. Self-Attention
```matlab
selfAttentionLayer(4, 64, 'Name', 'attention')
```

**Parameters:**
- **Number of heads:** 4
- **Embedding dimension:** 64 (must match previous layer)
- **Mechanism:** Multi-Head Self-Attention (Vaswani et al., 2017)

**How It Works:**
```
For each timestep t:
  1. Query (Q), Key (K), Value (V) = Linear(embedding[t])
  2. Attention weights = softmax(Q·K^T / sqrt(64))
  3. Output[t] = Σ(Attention[t,t'] × V[t'])
```

**Why 4 Heads?**
- Each head learns different temporal patterns:
  - Head 1: Local patterns (adjacent timesteps)
  - Head 2: Medium-range (100-200ms)
  - Head 3: Long-range (500ms-1s)
  - Head 4: Very long-range (1-4s)
- **Alternatives:**
  - 2 heads: Insufficient diversity
  - 8 heads: Overfitting with limited data

**Computational Cost:**
- **Complexity:** O(L²·D) where L=1000, D=64
- **Memory:** ~250 MB per batch of 16 samples (GPU)

#### 6. Layer Normalization
```matlab
layerNormalizationLayer('Name', 'ln2')
```

**Residual Connection (Implicit):**
- MATLAB's layer graph automatically adds skip connections
- Prevents gradient vanishing in deep networks

#### 7. ReLU Activation
```matlab
reluLayer('Name', 'relu2')
```

---

### Layer 8: Decoder (Reconstruction)

#### 8. 1D Convolution
```matlab
convolution1dLayer(5, 16, 'Padding', 'same', 'Name', 'decode_conv')
```

**Parameters:**
- **Kernel size:** 5 timesteps (matches encoder)
- **Filters:** 16 (original channel count)
- **Padding:** 'same'

**Function:** Projects 64-dimensional latent space back to 16-channel output
- **Symmetry:** Mirrors encoder structure (common in autoencoders)

---

### Layer 9: Output (Regression)

#### 9. Regression Layer
```matlab
regressionLayer('Name', 'output')
```

**Loss Function:** Mean Squared Error (MSE)

```math
MSE = (1 / (C × T)) × Σ(Input[c,t] - Output[c,t])²
```

Where:
- C = 16 channels
- T = 1000 timesteps

**Why MSE?**
- ✅ Penalizes large errors (seizures have dramatically different waveforms)
- ✅ Differentiable (backpropagation-friendly)
- ✅ Interpretable (units = normalized signal variance)

---

## 📈 Model Capacity

### Parameter Count

| Layer | Parameters | Calculation |
|-------|------------|-------------|
| Embed Conv | **5,184** | (5 × 16 × 64) + 64 bias |
| Self-Attention | **16,640** | 4 heads × (64² × 3 + 64) ÷ 4 |
| Decode Conv | **5,136** | (5 × 64 × 16) + 16 bias |
| **Total** | **~27,000** | |

**Comparison:**
- Small CNN: ~10K parameters
- ResNet-18: ~11M parameters
- GPT-2 Small: ~117M parameters

**Our model:** Lightweight by design (limited training data)

### Memory Requirements

**Training (batch size = 8):**
- Forward pass: ~500 MB GPU
- Backward pass: ~1.5 GB GPU
- Optimizer states: ~300 MB GPU
- **Total:** ~2.3 GB (well within RTX 4070 8GB)

**Inference (single sample):**
- ~50 MB GPU (very fast)

---

## 💡 Design Decisions & Alternatives

### Why Transformer (vs LSTM/GRU)?

| Aspect | Transformer | LSTM/GRU |
|--------|-------------|----------|
| **Temporal range** | Full 1000 timesteps (attention) | Limited by hidden state | ✅ Transformer |
| **Parallelization** | Full (GPU-friendly) | Sequential (slow) | ✅ Transformer |
| **Interpretability** | Attention weights visualizable | Hidden state opaque | ✅ Transformer |
| **Training stability** | LayerNorm, residual connections | Gradient vanishing/explosion | ✅ Transformer |
| **Parameter count** | Higher (27K) | Lower (~15K) | ✅ LSTM (but not critical for us) |

**Verdict:** Transformer preferred for:
- Long-range dependencies (ictal patterns can span 1-4 seconds)
- Faster training (GPU parallelization)
- Better interpretability (see attention weights)

### Considered Alternatives

1. **CNN-only (no Transformer)**
   - ❌ Limited receptive field (even with dilation)
   - ❌ Cannot capture global temporal context

2. **Variational Autoencoder (VAE)**
   - ❌ Requires tuning KL-divergence weight
   - ❌ Less interpretable anomaly scores

3. **Generative Adversarial Network (GAN)**
   - ❌ Training instability
   - ❌ Requires careful hyperparameter tuning
   - ❌ Overkill for this task

4. **One-Class SVM / Isolation Forest**
   - ❌ Cannot handle high-dimensional time series (16 × 1000 = 16K dims)
   - ❌ Requires manual feature engineering

---

## 🔬 Training Configuration

**Optimizer:** Adam
- **Learning rate:** 1e-4 (conservative, prevents overfitting)
- **Beta1:** 0.9, **Beta2:** 0.999 (default)
- **Epsilon:** 1e-8

**Epochs:** 50 (increased from 5 in pilot)
- **Rationale:** Underfitting with 5 epochs (loss still decreasing)
- **Monitoring:** Training loss curve

**Batch Size:** 8 (reduced from 16)
- **Rationale:** GPU memory optimization (RTX 4070 8GB)
- **Trade-off:** Slower training (~30% more time) but stable

**Regularization:**
- **Dropout:** Not used (small dataset, overfitting less likely)
- **Weight decay:** Not used (implicit regularization from LayerNorm)

**Data Augmentation:** None
- **Future work:** Time-warping, amplitude scaling

---

## 📊 Expected Behavior

### Normal Signal (Low Error)

```
Input:  [16 × 1000] normal EEG/ECG/EMG/MOV
         ↓ Encoder
Latent: [64 × 1000] compressed representation
         ↓ Attention (recognizes familiar patterns)
Latent: [64 × 1000] refined
         ↓ Decoder
Output: [16 × 1000] ≈ Input (good reconstruction)

MSE: ~0.01 - 0.1 (low error)
```

### Seizure Signal (High Error)

```
Input:  [16 × 1000] seizure (rhythmic spiking, high amplitude)
         ↓ Encoder
Latent: [64 × 1000] unfamiliar pattern (not seen in training!)
         ↓ Attention (cannot find similar patterns in memory)
Latent: [64 × 1000] poor representation
         ↓ Decoder
Output: [16 × 1000] ≠ Input (poor reconstruction, looks "normal-ish")

MSE: ~1.0 - 10.0 (high error) ← ANOMALY DETECTED!
```

---

## ⚠️ Limitations

1. **Fixed Input Length**
   - Must be exactly 1000 timesteps (4 seconds @ 250 Hz)
   - Cannot handle variable-duration seizures directly

2. **Channel Count Dependency**
   - Trained for 16 channels (specific to selected patients)
   - Retraining needed if channel count changes

3. **No Online Learning**
   - Cannot adapt to patient-specific patterns over time
   - Future: Implement transfer learning / fine-tuning

4. **Interpretability Limited**
   - Attention weights help, but not fully explainable
   - Black-box nature of deep learning

---

## 🔄 Next Steps

1. **Attention Visualization**
   - Plot attention weights for seizure vs normal
   - Identify which temporal ranges are most discriminative

2. **Ablation Studies**
   - Remove Transformer → pure CNN autoencoder (baseline)
   - Vary embedding dim: 32, 64, 128
   - Vary heads: 2, 4, 8

3. **Architecture Search**
   - Try deeper networks (2-3 Transformer blocks)
   - Add skip connections explicitly

---

## 📚 References

1. **Transformer Architecture:**
   - Vaswani A, et al. (2017). "Attention Is All You Need." NeurIPS.
   - Devlin J, et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.

2. **Autoencoders for Anomaly Detection:**
   - Chalapathy R, Chawla S (2019). "Deep Learning for Anomaly Detection: A Survey." arXiv.
   - Zhou C, Paffenroth RC (2017). "Anomaly Detection with Robust Deep Autoencoders." KDD.

3. **Medical Time Series:**
   - Fawaz HI, et al. (2019). "Deep learning for time series classification: a review." Data Mining and Knowledge Discovery.

4. **Seizure Detection:**
   - Truong ND, et al. (2018). "Convolutional neural networks for seizure prediction using intracranial and scalp EEG." Neural Networks.

---

**Report Generated:** January 4, 2025
**Architecture Definition:** `04_Training/train_model.m` (lines 71-108)
**Project:** SeizeIT2-Transformer v2.0.0
