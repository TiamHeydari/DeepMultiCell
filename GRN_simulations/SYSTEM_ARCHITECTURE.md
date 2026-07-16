# System Architecture & Understanding

## Project Overview

This project implements a **DeepSets Variational Autoencoder (VAE)** for analyzing Gene Regulatory Network (GRN) simulations on 3D retinal organoid data.

### Key Components

```
DeepMultiCell/
├── GRN_simulations/
│   ├── DeepSets_VAE_Jess_3geneData.ipynb  ← Main notebook
│   ├── model/
│   │   ├── __init__.py
│   │   ├── data_loading.py
│   │   ├── deepsets_vae.py
│   │   ├── models.py                 ← Model architectures
│   │   ├── training.py               ← UPDATED with moment loss
│   │   └── visualizations.py
│   └── data/
│       └── 3GeneResults_k2_1.h5ad
```

## Model Architectures (models.py)

The project uses several DeepSet autoencoder variants:

### DeepSet_Auto_encoder_v1
- Basic DeepSet layer with linear encoder/decoder
- Simplest architecture

### DeepSet_Auto_encoder_v2_SetTransformer
- Uses Set Attention Blocks (SAB)
- Multi-head attention for set learning

### DeepSet_Auto_encoder_v3_SetTransformer (Currently Used)
- Enhanced SAB with layer normalization
- Seed vectors for better expressiveness
- **Currently active in notebook**

### DeepSet_Auto_encoder_v4_SetTransformer
- Latest variant with additional enhancements

## Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Notebook (Jess_3geneData)             │
├─────────────────────────────────────────────────────────┤
│ 1. Load data (h5ad files)                               │
│ 2. Create DataLoaders (train/val/test/ood splits)      │
│ 3. Initialize model (v3 SetTransformer)                │
│ 4. Configure loss & optimization                        │
│ 5. Run fit_model()                                      │
│ 6. Analyze latent space & reconstructions              │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│               fit_model() in training.py                 │
├─────────────────────────────────────────────────────────┤
│ For each epoch:                                          │
│  1. Compute beta (KL weight schedule)                  │
│  2. Compute lambda_hungarian (Hungarian weight)        │
│  3. Compute lambda_moment (Moment weight)  ← NEW       │
│  4. train_epoch()                                       │
│  5. eval_epoch()                                        │
│  6. Save best model                                     │
│  7. Update history                                      │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│            train_epoch() / eval_epoch()                  │
├─────────────────────────────────────────────────────────┤
│ For each batch:                                          │
│  1. Forward pass: x_hat, z, mu, logvar = model(x)      │
│  2. Reconstruction loss:                               │
│     - Sinkhorn loss (optimal transport)                │
│     - Hungarian loss (point matching) [if λ > 0]      │
│     - Moment loss (statistical matching) [if λ > 0] ← NEW
│  3. KL divergence (VAE prior matching)                 │
│  4. Total: L = recon + β*KL                            │
│  5. Backprop (train only)                              │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│           reconstruction_loss() [UPDATED]                │
├─────────────────────────────────────────────────────────┤
│ loss_sink = sinkhorn_loss(x_hat, x)                     │
│                                                         │
│ if lambda_hungarian > 0:                               │
│     loss_hung = hungarian_loss(x_hat, x)              │
│ else:                                                   │
│     loss_hung = 0                                      │
│                                                         │
│ if lambda_moment > 0:                        ← NEW     │
│     loss_mom = moment_loss(x_hat, x)        ← NEW     │
│ else:                                        ← NEW     │
│     loss_mom = 0                             ← NEW     │
│                                                         │
│ loss_recon = loss_sink                                  │
│            + lambda_hungarian * loss_hung              │
│            + lambda_moment * loss_mom         ← NEW   │
│                                                         │
│ return (loss_recon, loss_sink, loss_hung, loss_mom) ← 4 VALUES
└─────────────────────────────────────────────────────────┘
```

## Loss Components Explained

### 1. Sinkhorn Loss (Always Active)
- Optimal transport distance between point clouds
- Differentiable approximation of Wasserstein distance
- Parameter: `blur` (trade-off between quality and smoothness)

### 2. Hungarian Loss (Conditional)
- 1-to-1 point matching using Hungarian algorithm
- Complements Sinkhorn with hard constraints
- Activated by: `lambda_hungarian > 0`
- Scheduled by: `hungarian_start_fraction`, `hungarian_ramp_fraction`

### 3. Moment Loss (NEW - Conditional)
- Statistical matching of moments (mean, variance, covariance)
- Preserves population-level properties
- Activated by: `lambda_moment_max > 0`
- Scheduled by: `moment_start_fraction`, `moment_ramp_fraction`

### 4. KL Loss (Always Active)
- Variational autoencoder prior matching
- Regularizes latent space to N(0, 1)
- Weight: `beta` (annealed during training)

## Data Flow

```
Input: h5ad file (3 genes × thousands of cells)
  ↓
Colony sampling (group by topology + colony)
  ↓
Point clouds (B=batch_size, N=colony_size, C=3_genes)
  ↓
Model encoding → z (B, latent_dim)
  ↓
Model decoding → x_hat (B, N, 3)
  ↓
Loss computation:
  ├─ Reconstruction: |x_hat - x|
  ├─ KL: D_KL(q(z|x) || p(z))
  └─ Moment: match statistics
  ↓
Backprop & weights update
```

## File Dependencies

```
notebook (DeepSets_VAE_Jess_3geneData.ipynb)
  ├─ imports training.py
  │  ├─ fit_model()         ← UPDATED
  │  ├─ train_epoch()       ← UPDATED
  │  ├─ eval_epoch()        ← UPDATED
  │  ├─ plot_training_history() ← UPDATED
  │  ├─ reconstruction_loss() ← UPDATED
  │  ├─ moment_loss()       ← NEW
  │  ├─ sinkhorn_loss()
  │  ├─ hungarian_loss()
  │  └─ ...
  │
  ├─ imports models.py
  │  ├─ DeepSet_Auto_encoder_v1
  │  ├─ DeepSet_Auto_encoder_v2_SetTransformer
  │  ├─ DeepSet_Auto_encoder_v3_SetTransformer ← Used
  │  └─ DeepSet_Auto_encoder_v4_SetTransformer
  │
  ├─ imports data_loading.py
  │  ├─ load_adata()
  │  ├─ downsample_adata_by_colony()
  │  ├─ ColonyFromAnnDataDataset
  │  ├─ build_dataloaders()
  │  └─ create_batch()
  │
  └─ imports visualizations.py
     ├─ pca_and_highlight_classes()
     ├─ umap_and_highlight_classes()
     ├─ show_reconstruction_examples()
     └─ ...
```

## Hyperparameter Flow

```
fit_model() parameters:
├─ n_epochs
├─ lr (learning rate)
├─ beta_start, beta_end, warmup_fraction
│  └─ Used by: get_beta_linear()
│
├─ blur (Sinkhorn smoothness)
│  └─ Used by: reconstruction_loss() → sinkhorn_loss()
│
├─ lambda_hungarian_max
├─ hungarian_start_fraction, hungarian_ramp_fraction
│  └─ Used by: get_lambda_hungarian() for Hungarian scheduling
│
├─ lambda_moment_max            ← NEW
├─ moment_start_fraction        ← NEW
├─ moment_ramp_fraction         ← NEW
│  └─ Used by: get_lambda_hungarian() for Moment scheduling ← NEW
│
└─ model_path, factor, patience, save_on
   └─ Used by: optimizer/scheduler setup & model checkpointing
```

## Scheduling System

All three loss components use schedules to control their contribution:

```
Epoch:     0    20    40    60    80    100
           |-----|-----|-----|-----|-----|

Beta:      0 ──→ 1e-5 ──→ 1e-5 ──→ 1e-5  (Linear then constant)

Lambda_HUN:  0 ─────────→ 0.1 ──────────→ 0.1  (Delayed ramp)

Lambda_MOM:  0 ────────→ 0.05 ────────→ 0.05  (Later delayed ramp)
```

## Model Training Phases

### Phase 1: Reconstruction Focus (Early epochs)
- Low beta: focus on reconstruction accuracy
- No Hungarian: only Sinkhorn
- No Moment: focus on point-wise matching

### Phase 2: Constraint Introduction (Middle epochs)
- Increasing beta: start regularizing latent space
- Ramping Hungarian: add point matching constraints
- Starting Moment: begin statistical matching

### Phase 3: Balanced Learning (Later epochs)
- Stable beta: maintain latent regularization
- Full Hungarian: complete point matching
- Full Moment: complete statistical matching

## Moment Loss Benefit for GRN Data

In GRN simulations:
- Individual cells (points) have specific states
- Colonies (sets) have population properties
- Moment loss ensures:
  - **Mean**: Average gene expression preserved
  - **Variance**: Gene expression heterogeneity preserved
  - **Covariance**: Gene-gene relationships preserved

This is biologically important because:
1. **Phenotype depends on population statistics**, not just individuals
2. **Cell types form due to regulatory relationships** (captured by covariance)
3. **Noise/heterogeneity is biological signal** (captured by variance)

## Computational Efficiency

Moment loss is computationally efficient:
- ✅ Only computes 3 statistics (mean, var, cov)
- ✅ Per-batch computation (no global stats needed)
- ✅ Low memory overhead
- ✅ Differentiable (standard PyTorch operations)

## Integration Summary

The moment loss integrates seamlessly because:

1. **Same scheduling mechanism** as Hungarian loss
2. **Same return structure** (4-tuple from reconstruction_loss)
3. **Same tracking** in history dictionary
4. **Same plotting** in visualization function
5. **No breaking changes** to existing code
6. **Optional by default** (lambda_moment_max=0.0)

---

**Key Innovation**: Moment loss adds statistical population-level matching to point-level reconstruction losses, making the VAE more suitable for biological set data where properties emerge at the population level.
