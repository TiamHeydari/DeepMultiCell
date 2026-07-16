# Example Usage of Moment Loss in Training

## Example 1: Baseline (No Moment Loss)
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

plot_training_history(history)
```

## Example 2: With Moment Loss (Light)
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
    # NEW: Moment Loss
    lambda_moment_max=0.02,
    moment_start_fraction=0.4,
    moment_ramp_fraction=0.2,
    model_path="model.pt",
    save_on="valid_recon",
)

plot_training_history(history)
```

## Example 3: With Moment Loss (Moderate)
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
    # NEW: Moment Loss
    lambda_moment_max=0.05,        # 5% weight on moment loss
    moment_start_fraction=0.3,     # Start at 30% of training
    moment_ramp_fraction=0.3,      # Ramp up over next 30%
    model_path="model.pt",
    save_on="valid_recon",
)

plot_training_history(history)
```

## Example 4: With Moment Loss (Strong)
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
    # NEW: Moment Loss
    lambda_moment_max=0.1,         # 10% weight on moment loss
    moment_start_fraction=0.0,     # Start immediately
    moment_ramp_fraction=0.2,      # Ramp up over first 20%
    model_path="model.pt",
    save_on="valid_recon",
)

plot_training_history(history)
```

## Understanding the Output

When moment loss is enabled, you'll see additional output:

```
Epoch: 1 | beta: 0.000000 | lambda_hungarian: 0.000000 | lambda_moment: 0.000000 | 
Train Loss: 0.123456 | Train Recon: 0.100000 | Train Sink: 0.050000 | Train Hung: 0.000000 | 
Train Mom: 0.000000 | Train KL: 0.023456 | Valid Loss: 0.145678 | Valid Recon: 0.120000 | 
Valid Sink: 0.055000 | Valid Hung: 0.000000 | Valid Mom: 0.000000 | Valid KL: 0.025678

Epoch: 50 | beta: 0.000001 | lambda_hungarian: 0.100000 | lambda_moment: 0.033333 | 
Train Loss: 0.098765 | Train Recon: 0.080000 | Train Sink: 0.040000 | Train Hung: 0.020000 | 
Train Mom: 0.020000 | Train KL: 0.018765 | Valid Loss: 0.112345 | Valid Recon: 0.090000 | 
Valid Sink: 0.042000 | Valid Hung: 0.018000 | Valid Mom: 0.025000 | Valid KL: 0.022345
```

Key metrics:
- **lambda_moment**: Increases as training progresses (0 → max value)
- **Train Mom**: Moment loss component (should decrease)
- **Valid Mom**: Validation moment loss (monitor for overfitting)

## Hyperparameter Tuning Tips

### To increase moment loss weight:
- Increase `lambda_moment_max` (e.g., 0.05 → 0.1)

### To start moment loss earlier:
- Decrease `moment_start_fraction` (e.g., 0.5 → 0.2)

### To ramp moment loss faster:
- Decrease `moment_ramp_fraction` (e.g., 0.3 → 0.1)

### To delay moment loss (let reconstruction stabilize first):
- Increase `moment_start_fraction` (e.g., 0.3 → 0.7)

## Comparing Models

To compare baseline vs. moment loss models:

```python
# Load baseline model
model_baseline = DeepSet_Auto_encoder_v3_SetTransformer(...)
model_baseline.load_state_dict(torch.load('model_baseline.pt'))

# Load moment loss model
model_moment = DeepSet_Auto_encoder_v3_SetTransformer(...)
model_moment.load_state_dict(torch.load('model_moment.pt'))

# Compare on test set
beta = 1e-3
loss_base, recon_base, kl_base, _, _, = test(model_baseline, device, test_loader, beta=beta)
loss_mom,  recon_mom,  kl_mom,  _, _, = test(model_moment, device, test_loader, beta=beta)

print(f"Baseline - Loss: {loss_base:.6f}, Recon: {recon_base:.6f}, KL: {kl_base:.6f}")
print(f"Moment   - Loss: {loss_mom:.6f}, Recon: {recon_mom:.6f}, KL: {kl_mom:.6f}")
```
