import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_samples(data_, label, marker="o"):
    if isinstance(data_, torch.Tensor):
        data_ = data_.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    plt.figure(figsize=(4, 4))
    plt.scatter(data_[:, 0], data_[:, 1], edgecolor="#333", marker=marker, label="Class " + str(label))
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.xlim([-1, 4])
    plt.ylim([-1, 4])
    plt.legend()
    plt.show()


def visualize_samples_3d(data_, label, marker="o"):
    if isinstance(data_, torch.Tensor):
        data_ = data_.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    if hasattr(label, "shape") and len(getattr(label, "shape", [])) > 0:
        if label.size == 1:
            label = label.item()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        data_[:, 0],
        data_[:, 1],
        data_[:, 2],
        marker=marker,
        label="Class " + str(label),
    )

    ax.set_title("Dataset samples")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$x_3$")
    ax.set_xlim([-0, 4])
    ax.set_ylim([-0, 4])
    ax.set_zlim([-0, 4])
    ax.legend()
    plt.show()


@torch.no_grad()
def show_reconstruction_examples(model, loader, device, n_show_examples=20, n_per_row=5):
    model.eval()

    x_list = []
    xhat_list = []
    len_list = []

    for batch in loader:
        x = batch["point"].to(device)
        seq_length = batch["seq_length"]
        x_hat, z, mu, logvar = model(x)
        x_list.append(x.cpu())
        xhat_list.append(x_hat.cpu())
        len_list.append(seq_length.cpu())

    x_all = torch.cat(x_list, dim=0).numpy()
    xhat_all = torch.cat(xhat_list, dim=0).numpy()
    seq_all = torch.cat(len_list, dim=0).numpy()

    np.random.seed(42)
    idx = np.random.choice(len(x_all), size=n_show_examples, replace=False)

    x_all = x_all[idx]
    xhat_all = xhat_all[idx]
    seq_all = seq_all[idx]

    fig = plt.figure(figsize=(12, 4 * n_show_examples))

    for i in range(n_show_examples):
        n_i = int(seq_all[i])

        ax1 = fig.add_subplot(n_show_examples, 3, 3 * i + 1)
        ax1.scatter(x_all[i, :n_i, 0], x_all[i, :n_i, 1], c="grey", alpha=0.6)
        ax1.scatter(xhat_all[i, :n_i, 0], xhat_all[i, :n_i, 1], c="blue", alpha=0.4)
        ax1.set_title(f"Example {i} (XY)")
        ax1.set_xlim(-1, 7)
        ax1.set_ylim(-1, 7)

        ax2 = fig.add_subplot(n_show_examples, 3, 3 * i + 2)
        ax2.scatter(x_all[i, :n_i, 0], x_all[i, :n_i, 2], c="grey", alpha=0.6)
        ax2.scatter(xhat_all[i, :n_i, 0], xhat_all[i, :n_i, 2], c="blue", alpha=0.4)
        ax2.set_title(f"Example {i} (XZ)")
        ax2.set_xlim(-1, 7)
        ax2.set_ylim(-1, 7)

        ax3 = fig.add_subplot(n_show_examples, 3, 3 * i + 3)
        ax3.scatter(x_all[i, :n_i, 1], x_all[i, :n_i, 2], c="grey", alpha=0.6)
        ax3.scatter(xhat_all[i, :n_i, 1], xhat_all[i, :n_i, 2], c="blue", alpha=0.4)
        ax3.set_title(f"Example {i} (YZ)")
        ax3.set_xlim(-1, 7)
        ax3.set_ylim(-1, 7)

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def collect_all_sets_z(model, loader, device, use_mu=True):
    model.eval()

    z_chunks = []
    y_chunks = []

    for batch in loader:
        x = batch["point"].to(device)
        y = batch["label"]
        x_hat, z, mu, logvar = model(x)
        latent = mu if use_mu else z
        z_chunks.append(latent.cpu())
        y_chunks.append(y.cpu())

    z_all = torch.cat(z_chunks, dim=0).numpy()
    y_all = torch.cat(y_chunks, dim=0).numpy()
    return z_all, y_all


def pca_and_highlight_classes(Z, y, pcx=1, pcy=2):
    pca = PCA()
    Z_pca = pca.fit_transform(Z)

    ix = pcx - 1
    iy = pcy - 1

    plt.figure(figsize=(7, 6))

    classes = np.unique(y)
    for cls in classes:
        mask = y == cls
        plt.scatter(Z_pca[mask, ix], Z_pca[mask, iy], alpha=0.8, s=18)

    plt.xlabel(f"PC{pcx}")
    plt.ylabel(f"PC{pcy}")
    plt.title(f"PCA of latent space: PC{pcx} vs PC{pcy}")
    plt.tight_layout()
    plt.show()


def umap_and_highlight_classes(
    Z,
    y,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42,
):
    try:
        import umap.umap_ as umap
    except ImportError as exc:
        raise ImportError("UMAP is not installed. Install it with: pip install umap-learn") from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    Z2 = reducer.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z2[:, 0], Z2[:, 1], s=8, alpha=0.12)

    classes = np.unique(y)
    for cls in classes:
        idx = y == cls
        if np.any(idx):
            plt.scatter(Z2[idx, 0], Z2[idx, 1], s=15, alpha=0.9, label=str(cls))

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP of autoencoder latent space")
    plt.legend(title="class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()








import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# Latent collection
# -----------------------------
model_for_latent = model if ("model" in globals() and hasattr(model, "eval")) else model_test

Z_train, y_train = collect_all_sets_z(model_for_latent, train_loader, device)
Z_val,   y_val   = collect_all_sets_z(model_for_latent, valid_loader, device)
Z_test,  y_test  = collect_all_sets_z(model_for_latent, test_loader, device)
Z_ood,   y_ood   = collect_all_sets_z(model_for_latent, ood_loader, device)

# -----------------------------
# PCA helpers
# -----------------------------
def fit_pca_on_reference(Z_ref, n_components=None):
    """
    Fit PCA on a reference set (usually train) and return:
      - fitted PCA object
      - transformed reference coordinates
    """
    pca = PCA(n_components=n_components)
    Z_ref_pca = pca.fit_transform(Z_ref)
    return pca, Z_ref_pca

def transform_with_pca(pca, Z):
    """
    Transform new data using an already-fitted PCA.
    """
    return pca.transform(Z)

# -----------------------------
# 2D PCA plot
# -----------------------------
def pca_plot_2d(
    Z_pca,
    y,
    pcx=1,
    pcy=2,
    title=None,
    figsize=(7, 6),
    s=22,
    alpha=0.8
):
    ix = pcx - 1
    iy = pcy - 1

    plt.figure(figsize=figsize)

    classes = np.unique(y)
    for cls in classes:
        mask = y == cls
        plt.scatter(
            Z_pca[mask, ix],
            Z_pca[mask, iy],
            s=s,
            alpha=alpha,
            label=str(cls)
        )

    plt.xlabel(f"PC{pcx}")
    plt.ylabel(f"PC{pcy}")
    plt.title(title if title is not None else f"PCA latent space: PC{pcx} vs PC{pcy}")
    plt.legend(title="Class", frameon=False)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 3D PCA plot
# -----------------------------
def pca_plot_3d(
    Z_pca,
    y,
    pcx=1,
    pcy=2,
    pcz=3,
    title=None,
    interactive=True,
    s=4
):
    ix = pcx - 1
    iy = pcy - 1
    iz = pcz - 1

    title = title if title is not None else f"PCA latent space: PC{pcx} vs PC{pcy} vs PC{pcz}"

    if interactive:
        try:
            import plotly.express as px

            # build a small dataframe-like dict
            data = {
                f"PC{pcx}": Z_pca[:, ix],
                f"PC{pcy}": Z_pca[:, iy],
                f"PC{pcz}": Z_pca[:, iz],
                "class": y.astype(str)
            }

            fig = px.scatter_3d(
                data,
                x=f"PC{pcx}",
                y=f"PC{pcy}",
                z=f"PC{pcz}",
                color="class",
                opacity=0.8,
                title=title
            )

            fig.update_traces(marker=dict(size=s))
            fig.update_layout(
                margin=dict(l=0, r=0, b=0, t=40),
                legend_title_text="Class"
            )
            fig.show()
            return

        except ImportError:
            print("plotly not found, falling back to matplotlib 3D.")

    # matplotlib fallback
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    classes = np.unique(y)
    for cls in classes:
        mask = y == cls
        ax.scatter(
            Z_pca[mask, ix],
            Z_pca[mask, iy],
            Z_pca[mask, iz],
            s=18,
            alpha=0.8,
            label=str(cls)
        )

    ax.set_xlabel(f"PC{pcx}")
    ax.set_ylabel(f"PC{pcy}")
    ax.set_zlabel(f"PC{pcz}")
    ax.set_title(title)
    ax.legend(title="Class")
    plt.tight_layout()
    plt.show()
