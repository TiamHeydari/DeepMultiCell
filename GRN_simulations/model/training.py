import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Normal, kl_divergence
from geomloss import SamplesLoss
from scipy.optimize import linear_sum_assignment


# =========================================================
# Losses
# =========================================================

def sinkhorn_loss(x_hat, x, blur=0.5, p=2):
    """
    x_hat : (B, N_pred, C)
    x     : (B, N_true, C)
    """
    loss_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    loss = loss_fn(x_hat, x)   # (B,)
    return loss.mean()


def hungarian_loss(x_hat, x, p=2):
    """
    Hungarian 1-to-1 matching loss for batched point sets.

    Parameters
    ----------
    x_hat : torch.Tensor
        Predicted point sets of shape (B, N_pred, C)
    x : torch.Tensor
        Target point sets of shape (B, N_true, C)
    p : int or float, default=2
        Norm used in torch.cdist to build the cost matrix

    Returns
    -------
    loss : scalar tensor
        Mean Hungarian matching loss over the batch
    """
    if x_hat.ndim != 3 or x.ndim != 3:
        raise ValueError(f"x_hat and x must both have shape (B, N, C), got {x_hat.shape} and {x.shape}")

    B1, N_pred, C1 = x_hat.shape
    B2, N_true, C2 = x.shape

    if B1 != B2:
        raise ValueError(f"Batch size mismatch: x_hat has {B1}, x has {B2}")
    if C1 != C2:
        raise ValueError(f"Feature dimension mismatch: x_hat has {C1}, x has {C2}")

    batch_losses = []

    for b in range(B1):
        cost = torch.cdist(x_hat[b], x[b], p=p)  # (N_pred, N_true)

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())

        row_ind = torch.as_tensor(row_ind, device=x_hat.device, dtype=torch.long)
        col_ind = torch.as_tensor(col_ind, device=x.device, dtype=torch.long)

        matched_pred = x_hat[b, row_ind]
        matched_true = x[b, col_ind]

        sample_loss = F.mse_loss(matched_pred, matched_true, reduction="mean")
        batch_losses.append(sample_loss)

    return torch.stack(batch_losses).mean()


def reconstruction_loss(
    x_hat,
    x,
    blur=0.5,
    p_sinkhorn=2,
    p_hungarian=2,
    lambda_hungarian=0.0,
):
    """
    Combined reconstruction loss:
        Sinkhorn + lambda_hungarian * Hungarian

    IMPORTANT:
    If lambda_hungarian == 0, Hungarian is not computed at all.
    """
    loss_sink = sinkhorn_loss(x_hat, x, blur=blur, p=p_sinkhorn)

    if lambda_hungarian > 0.0:
        loss_hung = hungarian_loss(x_hat, x, p=p_hungarian)
        loss_recon = loss_sink + lambda_hungarian * loss_hung
    else:
        loss_hung = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)
        loss_recon = loss_sink

    return loss_recon, loss_sink, loss_hung


# =========================================================
# Optimizer / scheduler
# =========================================================

def make_optimizer_scheduler(model, lr=1e-4, factor=0.5, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
    )
    return optimizer, scheduler


# =========================================================
# Beta schedule
# =========================================================

def get_beta_linear(epoch, n_epochs, beta_start=1e-4, beta_end=1e-2, warmup_fraction=0.5):
    warmup_epochs = max(1, int(n_epochs * warmup_fraction))

    if epoch >= warmup_epochs:
        return beta_end

    alpha = epoch / (warmup_epochs - 1) if warmup_epochs > 1 else 1.0
    return beta_start + alpha * (beta_end - beta_start)


# =========================================================
# Hungarian schedule
# =========================================================

def get_lambda_hungarian(
    epoch,
    n_epochs,
    lambda_hungarian_max=0.1,
    hungarian_start_fraction=0.2,
    hungarian_ramp_fraction=0.3,
):
    """
    Delayed linear schedule for Hungarian weight.

    Default behavior:
    - first 20% of epochs: Sinkhorn only
    - next 30% of epochs: linearly ramp Hungarian from 0 to lambda_hungarian_max
    - remaining epochs: keep lambda_hungarian_max

    Example with n_epochs=20:
    - epochs 0-3   : lambda = 0
    - epochs 4-9   : ramp up
    - epochs 10-19 : fixed at max
    """
    if not (0.0 <= hungarian_start_fraction < 1.0):
        raise ValueError("hungarian_start_fraction must be in [0,1).")
    if not (0.0 <= hungarian_ramp_fraction <= 1.0):
        raise ValueError("hungarian_ramp_fraction must be in [0,1].")

    start_epoch = int(n_epochs * hungarian_start_fraction)
    ramp_epochs = max(1, int(n_epochs * hungarian_ramp_fraction))
    ramp_end_epoch = start_epoch + ramp_epochs

    if epoch < start_epoch:
        return 0.0
    elif epoch >= ramp_end_epoch:
        return lambda_hungarian_max
    else:
        alpha = (epoch - start_epoch) / max(1, ramp_epochs - 1)
        return alpha * lambda_hungarian_max


# =========================================================
# Train / eval
# =========================================================

def train_epoch(
    model,
    device,
    train_loader,
    optimizer,
    beta=1e-3,
    blur=0.5,
    lambda_hungarian=0.0,
):
    train_loss_ep = 0.0
    train_recon_ep = 0.0
    train_sink_ep = 0.0
    train_hung_ep = 0.0
    train_kl_ep = 0.0

    model.train()

    for batch in train_loader:
        data = batch["point"].to(device)
        optimizer.zero_grad()

        x_hat, z, mu, logvar = model(data)

        recon_loss, sink_loss, hung_loss = reconstruction_loss(
            x_hat=x_hat,
            x=data,
            blur=blur,
            p_sinkhorn=2,
            p_hungarian=2,
            lambda_hungarian=lambda_hungarian,
        )

        std = torch.exp(0.5 * logvar)
        q = Normal(mu, std)
        p = Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl_loss = kl_divergence(q, p).sum(dim=1).mean()

        loss = recon_loss + beta * kl_loss
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        train_loss_ep += loss.item() * batch_size
        train_recon_ep += recon_loss.item() * batch_size
        train_sink_ep += sink_loss.item() * batch_size
        train_hung_ep += hung_loss.item() * batch_size
        train_kl_ep += kl_loss.item() * batch_size

    n_samples = len(train_loader.dataset)
    train_loss_ep /= n_samples
    train_recon_ep /= n_samples
    train_sink_ep /= n_samples
    train_hung_ep /= n_samples
    train_kl_ep /= n_samples

    return train_loss_ep, train_recon_ep, train_sink_ep, train_hung_ep, train_kl_ep


@torch.no_grad()
def eval_epoch(
    model,
    device,
    loader,
    beta=1e-3,
    blur=0.5,
    lambda_hungarian=0.0,
):
    valid_loss_ep = 0.0
    valid_recon_ep = 0.0
    valid_sink_ep = 0.0
    valid_hung_ep = 0.0
    valid_kl_ep = 0.0

    model.eval()

    for batch in loader:
        data = batch["point"].to(device)
        x_hat, z, mu, logvar = model(data)

        recon_loss, sink_loss, hung_loss = reconstruction_loss(
            x_hat=x_hat,
            x=data,
            blur=blur,
            p_sinkhorn=2,
            p_hungarian=2,
            lambda_hungarian=lambda_hungarian,
        )

        std = torch.exp(0.5 * logvar)
        q = Normal(mu, std)
        p = Normal(torch.zeros_like(mu), torch.ones_like(std))
        kl_loss = kl_divergence(q, p).sum(dim=1).mean()

        loss = recon_loss + beta * kl_loss

        batch_size = data.size(0)
        valid_loss_ep += loss.item() * batch_size
        valid_recon_ep += recon_loss.item() * batch_size
        valid_sink_ep += sink_loss.item() * batch_size
        valid_hung_ep += hung_loss.item() * batch_size
        valid_kl_ep += kl_loss.item() * batch_size

    n_samples = len(loader.dataset)
    valid_loss_ep /= n_samples
    valid_recon_ep /= n_samples
    valid_sink_ep /= n_samples
    valid_hung_ep /= n_samples
    valid_kl_ep /= n_samples

    return valid_loss_ep, valid_recon_ep, valid_sink_ep, valid_hung_ep, valid_kl_ep


# =========================================================
# Fit
# =========================================================

def fit_model(
    model,
    device,
    train_loader,
    valid_loader,
    n_epochs=20,
    lr=1e-4,
    beta_start=1e-5,
    beta_end=5e-3,
    warmup_fraction=0.7,
    blur=0.5,
    lambda_hungarian_max=0.1,
    hungarian_start_fraction=0.2,   # default: first 20% only Sinkhorn
    hungarian_ramp_fraction=0.3,    # then ramp Hungarian
    model_path="model.pt",
    factor=0.5,
    patience=3,
    save_on="valid_loss",           # "valid_loss" or "valid_recon"
):
    optimizer, scheduler = make_optimizer_scheduler(model, lr=lr, factor=factor, patience=patience)

    if save_on not in {"valid_loss", "valid_recon"}:
        raise ValueError("save_on must be 'valid_loss' or 'valid_recon'.")

    best_metric = np.inf

    history = {
        "betas": [],
        "lambda_hungarians": [],
        "train_losses": [],
        "train_recon_losses": [],
        "train_sink_losses": [],
        "train_hung_losses": [],
        "train_kl_losses": [],
        "valid_losses": [],
        "valid_recon_losses": [],
        "valid_sink_losses": [],
        "valid_hung_losses": [],
        "valid_kl_losses": [],
    }

    for epoch in range(n_epochs):
        beta = get_beta_linear(
            epoch=epoch,
            n_epochs=n_epochs,
            beta_start=beta_start,
            beta_end=beta_end,
            warmup_fraction=warmup_fraction,
        )

        lambda_hungarian = get_lambda_hungarian(
            epoch=epoch,
            n_epochs=n_epochs,
            lambda_hungarian_max=lambda_hungarian_max,
            hungarian_start_fraction=hungarian_start_fraction,
            hungarian_ramp_fraction=hungarian_ramp_fraction,
        )

        history["betas"].append(beta)
        history["lambda_hungarians"].append(lambda_hungarian)

        train_loss, train_recon, train_sink, train_hung, train_kl = train_epoch(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            beta=beta,
            blur=blur,
            lambda_hungarian=lambda_hungarian,
        )

        valid_loss, valid_recon, valid_sink, valid_hung, valid_kl = eval_epoch(
            model=model,
            device=device,
            loader=valid_loader,
            beta=beta,
            blur=blur,
            lambda_hungarian=lambda_hungarian,
        )

        history["train_losses"].append(train_loss)
        history["train_recon_losses"].append(train_recon)
        history["train_sink_losses"].append(train_sink)
        history["train_hung_losses"].append(train_hung)
        history["train_kl_losses"].append(train_kl)

        history["valid_losses"].append(valid_loss)
        history["valid_recon_losses"].append(valid_recon)
        history["valid_sink_losses"].append(valid_sink)
        history["valid_hung_losses"].append(valid_hung)
        history["valid_kl_losses"].append(valid_kl)

        print(
            f"Epoch: {epoch + 1} | "
            f"beta: {beta:.6f} | "
            f"lambda_hungarian: {lambda_hungarian:.6f} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Recon: {train_recon:.6f} | "
            f"Train Sink: {train_sink:.6f} | "
            f"Train Hung: {train_hung:.6f} | "
            f"Train KL: {train_kl:.6f} | "
            f"Valid Loss: {valid_loss:.6f} | "
            f"Valid Recon: {valid_recon:.6f} | "
            f"Valid Sink: {valid_sink:.6f} | "
            f"Valid Hung: {valid_hung:.6f} | "
            f"Valid KL: {valid_kl:.6f}"
        )

        current_metric = valid_loss if save_on == "valid_loss" else valid_recon

        if current_metric <= best_metric:
            print(f"Validation {save_on} decreased ({best_metric:.6f} --> {current_metric:.6f}). Saving model ...")
            torch.save(model.state_dict(), model_path)
            best_metric = current_metric

        scheduler.step(valid_recon)

    return history


# =========================================================
# Aliases
# =========================================================

train = train_epoch
test = eval_epoch


# =========================================================
# Plotting
# =========================================================

def plot_training_history(history):
    epochs = range(1, len(history["betas"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_losses"], label="Train total loss")
    plt.plot(epochs, history["valid_losses"], label="Validation total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Total Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_recon_losses"], label="Train recon loss")
    plt.plot(epochs, history["valid_recon_losses"], label="Validation recon loss")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_sink_losses"], label="Train Sinkhorn")
    plt.plot(epochs, history["valid_sink_losses"], label="Validation Sinkhorn")
    plt.xlabel("Epoch")
    plt.ylabel("Sinkhorn Loss")
    plt.title("Sinkhorn Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_hung_losses"], label="Train Hungarian")
    plt.plot(epochs, history["valid_hung_losses"], label="Validation Hungarian")
    plt.xlabel("Epoch")
    plt.ylabel("Hungarian Loss")
    plt.title("Hungarian Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_kl_losses"], label="Train KL loss")
    plt.plot(epochs, history["valid_kl_losses"], label="Validation KL loss")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("KL Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["betas"], label="beta")
    plt.xlabel("Epoch")
    plt.ylabel("Beta")
    plt.title("Beta Annealing Schedule")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["lambda_hungarians"], label="lambda_hungarian")
    plt.xlabel("Epoch")
    plt.ylabel("Hungarian Weight")
    plt.title("Hungarian Weight Schedule")
    plt.legend()
    plt.show()