# Moment Loss Integration Summary

## Overview
Added a new **moment loss** function to the training pipeline that matches mean, variance, and covariance statistics between predicted and target point clouds.

## Changes Made

### 1. New Loss Function: `moment_loss()`
Located in [model/training.py](model/training.py)

```python
def moment_loss(x_hat, x, eps=1e-8):
    """
    Match mean, variance, and covariance of each colony.
    
    Parameters
    ----------
    x_hat : (B, N, C) - predicted point sets
    x     : (B, N, C) - target point sets
    eps   : float - numerical stability
    
    Returns
    -------
    loss : scalar - sum of mean, variance, and covariance losses
    """
```

**What it does:**
- Computes mean loss: MSE between predicted and target means
- Computes variance loss: MSE between predicted and target variances
- Computes covariance loss: MSE between predicted and target covariance matrices
- Returns sum of all three components

### 2. Updated `reconstruction_loss()`
Now accepts `lambda_moment` parameter and integrates moment loss:

```python
loss_recon = loss_sink + lambda_hungarian * loss_hung + lambda_moment * loss_mom
```

### 3. Updated Training Functions
- `train_epoch()`: Now tracks and returns moment loss
- `eval_epoch()`: Now tracks and returns moment loss  
- Both functions accept `lambda_moment` parameter

### 4. Updated `fit_model()` Function
New parameters added:

```python
lambda_moment_max=0.0,           # Maximum weight for moment loss (default: disabled)
moment_start_fraction=0.5,       # When to start ramping moment loss (fraction of total epochs)
moment_ramp_fraction=0.3,        # Duration to ramp moment loss (fraction of total epochs)
```

### 5. Updated History Dictionary
Now tracks moment loss components:
- `train_mom_losses`: Training moment loss per epoch
- `valid_mom_losses`: Validation moment loss per epoch
- `lambda_moments`: Moment loss weight schedule per epoch

### 6. Updated Plotting
`plot_training_history()` now includes a new plot for moment loss over time

## How to Use in Your Notebook

### Basic Usage (Moment Loss Disabled)
Simply call `fit_model()` as before - moment loss is disabled by default (`lambda_moment_max=0.0`):

```python
history = fit_model(
    model=model,
    device=device,
    train_loader=train_loader,
    valid_loader=valid_loader,
    n_epochs=100,
    lr=1e-4,
    beta_start=0.0,
    beta_end=1e-5,
    warmup_fraction=0.7,
    blur=0.05,
    lambda_hungarian_max=0.1,
    hungarian_start_fraction=0.2,
    hungarian_ramp_fraction=0.2,
    model_path="model.pt",
    save_on="valid_recon",
)
```

### With Moment Loss Enabled
Add the moment loss parameters:

```python
history = fit_model(
    model=model,
    device=device,
    train_loader=train_loader,
    valid_loader=valid_loader,
    n_epochs=100,
    lr=1e-4,
    
    # Beta schedule
    beta_start=0.0,
    beta_end=1e-5,
    warmup_fraction=0.7,
    
    # Sinkhorn + Hungarian
    blur=0.05,
    lambda_hungarian_max=0.1,
    hungarian_start_fraction=0.2,
    hungarian_ramp_fraction=0.2,
    
    # NEW: Moment loss
    lambda_moment_max=0.05,        # Maximum weight (e.g., 0.01 to 0.1)
    moment_start_fraction=0.5,     # Start at 50% of training
    moment_ramp_fraction=0.2,      # Ramp up over next 20% of training
    
    model_path="model.pt",
    save_on="valid_recon",
)
```

## Example Training Schedules

### Conservative Approach (Keep Statistical Matching Light)
```python
lambda_moment_max=0.01,
moment_start_fraction=0.7,
moment_ramp_fraction=0.2,
```
- Starts moment loss at 70% of training
- Only adds 1% weight to reconstruction
- Good for fine-tuning

### Aggressive Approach (Strong Statistical Matching)
```python
lambda_moment_max=0.1,
moment_start_fraction=0.0,
moment_ramp_fraction=0.3,
```
- Starts moment loss immediately
- Reaches 10% weight during first 30% of training
- Good if statistical properties are critical

### Balanced Approach (Recommended)
```python
lambda_moment_max=0.05,
moment_start_fraction=0.3,
moment_ramp_fraction=0.3,
```
- Starts at 30% of training
- Reaches 5% weight by 60% of training
- Maintains throughout rest of training
- Good balance between reconstruction and statistics

## Monitoring Training

The training output now includes moment loss:

```
Epoch: 1 | beta: 0.000000 | lambda_hungarian: 0.000000 | lambda_moment: 0.000000 | 
Train Loss: 0.123456 | Train Recon: 0.100000 | Train Sink: 0.050000 | Train Hung: 0.000000 | 
Train Mom: 0.000000 | Train KL: 0.023456 | Valid Loss: 0.145678 | Valid Recon: 0.120000 | 
Valid Sink: 0.055000 | Valid Hung: 0.000000 | Valid Mom: 0.000000 | Valid KL: 0.025678
```

### Plot Training History
```python
plot_training_history(history)
```

Now includes:
1. Total loss
2. Reconstruction loss
3. Sinkhorn loss
4. Hungarian loss
5. **Moment loss** ← New!
6. KL loss
7. Beta schedule
8. Hungarian weight schedule
9. **Moment weight schedule** ← New!

## Why Moment Loss?

The moment loss helps your model learn:
- **Statistical properties**: Mean, variance, covariance of gene expression
- **Population-level patterns**: Beyond individual point matching
- **Multivariate relationships**: Correlations between genes

This is especially useful for:
- GRN (Gene Regulatory Network) simulations
- Preserving biological plausibility
- Capturing colony-level characteristics

## Technical Details

### Moment Loss Computation
```
For each batch sample (colony):
  1. Compute sample means: mean(X), mean(Xhat)
  2. Center data: X_centered = X - mean(X)
  3. Compute variances: var(X_centered), var(Xhat_centered)
  4. Compute covariances: Cov = X_centered^T @ X_centered / N
  5. Loss = MSE(mean) + MSE(var) + MSE(cov)
```

### Scheduling
Uses same `get_lambda_hungarian()` function for consistency:
- Linear warmup from 0 to `lambda_moment_max`
- Controlled by `moment_start_fraction` and `moment_ramp_fraction`
- Once ramped up, stays at `lambda_moment_max` for remaining training

## Backward Compatibility

✅ **Fully backward compatible!** All existing code continues to work unchanged:
- Default `lambda_moment_max=0.0` disables moment loss
- Existing parameters unchanged
- New parameters are optional

## Files Modified

1. **[model/training.py](model/training.py)**
   - Added `moment_loss()` function
   - Updated `reconstruction_loss()`
   - Updated `train_epoch()`, `eval_epoch()`, `fit_model()`
   - Updated `plot_training_history()`

## Next Steps

1. Run training with `lambda_moment_max=0.0` (baseline)
2. Experiment with small values (`0.01-0.05`)
3. Monitor validation loss and moment loss component
4. Adjust `moment_start_fraction` and `moment_ramp_fraction` based on results
5. Compare final models to see if statistical matching improves biological validity

