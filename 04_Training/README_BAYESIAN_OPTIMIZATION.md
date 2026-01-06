# Bayesian Optimization Integration

## Overview
train_model.m now supports **automatic Bayesian hyperparameter optimization** before training. This feature finds optimal hyperparameters for your specific dataset automatically.

## Quick Start

### Option 1: Standard Training (Fast - 2-3 hours)
Uses pre-optimized parameters from config.m
```matlab
[net, trainInfo] = train_model();
```

### Option 2: With Bayesian Optimization (Slow - 4-5 hours)
Finds optimal hyperparameters first, then trains
```matlab
[net, trainInfo] = train_model('UseBayesianOpt', true);
```

### Option 3: Custom Optimization Iterations
More iterations = better optimization but slower
```matlab
% Quick optimization (10 iterations)
[net, trainInfo] = train_model('UseBayesianOpt', true, 'BayesOptIterations', 10);

% Thorough optimization (30 iterations)
[net, trainInfo] = train_model('UseBayesianOpt', true, 'BayesOptIterations', 30);
```

## How It Works

### Workflow

```
UseBayesianOpt = true
        ↓
┌───────────────────┐
│ STEP 1:           │
│ Run Bayesian Opt  │  ← Find optimal hyperparameters
│ (1-2 hours)       │
└───────────────────┘
        ↓
┌───────────────────┐
│ STEP 2:           │
│ Update Config     │  ← Apply optimal parameters
│ (automatic)       │
└───────────────────┘
        ↓
┌───────────────────┐
│ STEP 3:           │
│ Train Model       │  ← Train with optimal settings
│ (2-3 hours)       │
└───────────────────┘
        ↓
    Best Model
```

### What Gets Optimized?

The following hyperparameters are automatically tuned:

| Parameter | Search Range | Description |
|-----------|--------------|-------------|
| **Learning Rate** | [1e-5, 1e-2] | Step size for gradient descent |
| **Embedding Dim** | [32, 256] | Size of latent representation |
| **Attention Heads** | [2, 8] | Number of parallel attention mechanisms |
| **Encoder Layers** | [2, 6] | Depth of encoder network |
| **Decoder Layers** | [2, 6] | Depth of decoder network |
| **Dropout Rate** | [0.1, 0.5] | Regularization strength |
| **Batch Size** | [16, 128] | Number of samples per batch |
| **FFN Multiplier** | [1, 4] | Feed-forward network size multiplier |

### Optimization Strategy

- **Algorithm**: Bayesian Optimization with Gaussian Process
- **Objective**: Minimize reconstruction error on validation set
- **Cross-Validation**: 3-fold patient-wise CV
- **GPU Accelerated**: Yes (if available)

## Example Output

### Standard Training
```
=== TRANSFORMER AUTOENCODER TRAINING ===
Configuration: SeizeIT2-Transformer v2.0.0
Bayesian Optimization: DISABLED (using config parameters)

Training Parameters:
  Max Epochs: 50
  Batch Size: 32
  Learning Rate: 5.30e-04
  Hardware: RTX 4070 (8GB)

Loading training data...
```

### With Bayesian Optimization
```
=== TRANSFORMER AUTOENCODER TRAINING ===
Configuration: SeizeIT2-Transformer v2.0.0

========== BAYESIAN OPTIMIZATION ==========
Running hyperparameter optimization...
Iterations: 10
This may take 1-2 hours...

Iteration 1/10: Objective = 45.2341
Iteration 2/10: Objective = 38.1234
...
Iteration 10/10: Objective = 22.9641

========== OPTIMIZATION COMPLETE ==========
Optimal hyperparameters found:
  Learning Rate:      0.000530
  Embedding Dim:      186
  Attention Heads:    6
  Encoder Layers:     4
  Decoder Layers:     6
  Dropout Rate:       0.3606
  Batch Size:         117
  FFN Multiplier:     3.17
  Objective (Error):  22.9641

Training Parameters:
  Max Epochs: 50
  Batch Size: 117
  Learning Rate: 5.30e-04
  Hardware: RTX 4070 (8GB)

Loading training data...
```

## Saved Results

### Model Files

**With Bayesian Optimization:**
```
Data/ModelData/Models/
├── Trained_Transformer_BayesOpt_2026-01-05_14-30-45.mat  ← Includes "BayesOpt" in name
└── Trained_Transformer_Latest.mat
```

**Standard Training:**
```
Data/ModelData/Models/
├── Trained_Transformer_2026-01-05_14-30-45.mat
└── Trained_Transformer_Latest.mat
```

### Optimization Results

```
Results/BayesianOpt_AutoTrain/
└── optimization_results.mat  ← Detailed optimization history
```

### Model Metadata

All trained models include `training_metadata`:

```matlab
load('Data/ModelData/Models/Trained_Transformer_Latest.mat');

% Check if Bayesian optimization was used
if training_metadata.used_bayesopt
    fprintf('Model trained with Bayesian optimization\n');
    fprintf('Optimization iterations: %d\n', training_metadata.bayesopt_iterations);
else
    fprintf('Model trained with config parameters\n');
end

fprintf('Final learning rate: %.6f\n', training_metadata.final_learning_rate);
fprintf('Final batch size: %d\n', training_metadata.final_batch_size);
```

## Performance Comparison

| Mode | Training Time | Expected AUC | When to Use |
|------|---------------|--------------|-------------|
| **Standard** | ~2-3 hours | 0.91-0.93 | Quick experiments, baseline |
| **BayesOpt (10 iter)** | ~4-5 hours | 0.93-0.95 | Better performance needed |
| **BayesOpt (30 iter)** | ~6-8 hours | 0.94-0.96 | Maximum performance |

## Advanced Usage

### Combine with Other Parameters

```matlab
% Bayesian optimization + custom epochs + custom batch size override
[net, trainInfo] = train_model(...
    'UseBayesianOpt', true, ...
    'BayesOptIterations', 20, ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 16);  % This will be overridden by BayesOpt result
```

**Note**: If you specify `MiniBatchSize` or `InitialLearningRate`, they will be **overridden** by Bayesian optimization results.

### Resume From Previous Optimization

If you want to use previously found optimal parameters without re-running optimization:

1. Check archived results: `Results/Archive_*/BayesianOpt_*`
2. Load the best parameters manually
3. Update config.m
4. Run standard training

## Troubleshooting

### Error: "bayesian_optimization.m not found"

**Solution**: Make sure the path is added:
```matlab
addpath('03_Models');
[net, trainInfo] = train_model('UseBayesianOpt', true);
```

### Optimization Takes Too Long

**Solution 1**: Reduce iterations
```matlab
[net, trainInfo] = train_model('UseBayesianOpt', true, 'BayesOptIterations', 5);
```

**Solution 2**: Use previously optimized parameters (disable BayesOpt)
```matlab
[net, trainInfo] = train_model();  % Uses config.m parameters
```

### GPU Memory Error During Optimization

**Solution**: Edit `config.m` to reduce search space upper bounds:
```matlab
cfg.bayesopt.search_space.batch_size = [16, 64];  % Reduce from [16, 128]
cfg.bayesopt.search_space.embedding_dim = [32, 128];  % Reduce from [32, 256]
```

## Best Practices

### 1. First Run: Standard Training
```matlab
[net, trainInfo] = train_model();
```
- Get baseline results
- Check if current parameters are sufficient
- Faster iteration during development

### 2. Final Model: Bayesian Optimization
```matlab
[net, trainInfo] = train_model('UseBayesianOpt', true, 'BayesOptIterations', 20);
```
- Use for final model before publication
- Ensure optimal performance
- Document optimization results

### 3. Quick Experiments: Disable Optimization
```matlab
[net, trainInfo] = train_model('MaxEpochs', 10);  % Fast testing
```

## See Also

- **03_Models/bayesian_optimization.m** - Standalone optimization function
- **config/config.m** - Configuration and hyperparameter search space
- **04_Training/train_model.m** - Main training script
- **Results/BayesianOpt_AutoTrain/** - Saved optimization results
