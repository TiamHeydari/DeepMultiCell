from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils import data
from torch.utils.data import Subset


def load_adata(address: str):
    return sc.read_h5ad(address)


def downsample_adata_by_colony(adata, frac=1 / 3, group_cols=("topology_id", "colony_id"), random_state=42):
    obs = adata.obs.copy()

    missing = [c for c in group_cols if c not in obs.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns in adata.obs: {missing}")

    colony_df = obs.loc[:, list(group_cols)].drop_duplicates().reset_index(drop=True)

    n_colonies = len(colony_df)
    if n_colonies == 0:
        raise ValueError("No colonies found.")

    n_keep = max(1, int(round(frac * n_colonies)))

    rng = np.random.default_rng(random_state)
    keep_idx = rng.choice(n_colonies, size=n_keep, replace=False)
    kept_colonies = colony_df.iloc[keep_idx].copy()

    obs_keys = pd.MultiIndex.from_frame(obs.loc[:, list(group_cols)])
    keep_keys = pd.MultiIndex.from_frame(kept_colonies)
    mask = obs_keys.isin(keep_keys)

    adata_small = adata[mask].copy()

    print(f"Original cells: {adata.n_obs}")
    print(f"Downsampled cells: {adata_small.n_obs}")
    print(f"Original colonies: {n_colonies}")
    print(f"Kept colonies: {n_keep}")

    return adata_small


class ColonyFromAnnDataDataset(data.Dataset):
    """One sample is one colony extracted from an AnnData object."""

    def __init__(
        self,
        adata,
        time_point=250,
        gene_cols=("g1", "g2", "g3"),
        group_cols=("topology_id", "colony_id"),
        extra_meta_cols=("param_set_id", "cell_id", "k1", "k2", "signal_s1", "signal_s2", "signal_s3"),
        dtype=torch.float32,
    ):
        self.dtype = dtype
        self.gene_cols = list(gene_cols)
        self.group_cols = list(group_cols)

        mask = adata.obs["time"].round().to_numpy() == time_point
        ad = adata[mask].copy()

        if ad.n_obs == 0:
            raise ValueError(f"No cells found with time == {time_point}")

        X = ad[:, self.gene_cols].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        else:
            X = np.asarray(X)

        if X.shape[1] != len(self.gene_cols):
            raise ValueError(f"Expected {len(self.gene_cols)} gene columns, got shape {X.shape}")

        self.X = X.astype(np.float32, copy=False)
        obs = ad.obs.reset_index(drop=True).copy()

        grouped = obs.groupby(self.group_cols, sort=False, observed=True).indices
        self.group_keys = list(grouped.keys())
        self.group_indices = list(grouped.values())

        self.meta_list = []
        self.labels = []

        for key, idx in zip(self.group_keys, self.group_indices):
            rows = obs.iloc[idx]
            meta = {}

            if len(self.group_cols) == 1:
                meta[self.group_cols[0]] = key
            else:
                for c, v in zip(self.group_cols, key):
                    meta[c] = v

            for col in extra_meta_cols:
                vals = rows[col].to_numpy()
                if len(vals) == 0:
                    continue
                if np.all(vals == vals[0]):
                    meta[col] = vals[0]
                else:
                    meta[col] = None

            if "topology_id" not in meta:
                raise ValueError(
                    f"'topology_id' was not found in meta for colony key {key}. Check group_cols={self.group_cols}."
                )

            label = int(meta["topology_id"])
            self.meta_list.append(meta)
            self.labels.append(label)

        self.topology_params = ad.uns.get("topology_params", None)

        print(f"Filtered cells at time={time_point}: {ad.n_obs}")
        print(f"Number of colonies: {len(self.group_indices)}")

    def __len__(self):
        return len(self.group_indices)

    def __getitem__(self, idx):
        cell_idx = self.group_indices[idx]
        pts = torch.tensor(self.X[cell_idx], dtype=self.dtype)

        return {
            "point": pts,
            "label": self.labels[idx],
            "seq_length": pts.shape[0],
            "meta": self.meta_list[idx],
        }


def create_batch(batch, K=None):
    points = [item["point"] for item in batch]
    labels = [item["label"] for item in batch]
    metas = [item["meta"] for item in batch]
    lengths = torch.tensor([p.shape[0] for p in points], dtype=torch.long)

    if K is None:
        K = int(lengths.max().item())

    padded = []
    new_lengths = []

    for p in points:
        n = p.shape[0]

        if n > K:
            idx = torch.randperm(n)[:K]
            p2 = p[idx]
            n2 = K
        elif n < K:
            pad = torch.zeros(K - n, p.shape[1], dtype=p.dtype)
            p2 = torch.cat([p, pad], dim=0)
            n2 = n
        else:
            p2 = p
            n2 = n

        padded.append(p2)
        new_lengths.append(n2)

    x = torch.stack(padded, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    new_lengths = torch.tensor(new_lengths, dtype=torch.long)

    return {
        "point": x,
        "label": y,
        "seq_length": new_lengths,
        "meta": metas,
    }


def split_indices_by_topology(pc_dataset, seed=42, ood_fraction=0.1, valid_fraction=0.1, test_fraction=0.1):
    topology_to_indices = defaultdict(list)

    for index in range(len(pc_dataset)):
        topology = pc_dataset[index]["label"]
        topology_to_indices[topology].append(index)

    topologies = list(topology_to_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(topologies)

    n_total = len(topologies)
    n_ood = int(ood_fraction * n_total)
    n_val = int(valid_fraction * n_total)
    n_test = int(test_fraction * n_total)
    n_train = n_total - n_ood - n_val - n_test

    ood_topos = topologies[:n_ood]
    train_topos = topologies[n_ood : n_ood + n_train]
    val_topos = topologies[n_ood + n_train : n_ood + n_train + n_val]
    test_topos = topologies[n_ood + n_train + n_val :]

    train_indices = []
    valid_indices = []
    test_indices = []
    ood_indices = []

    for topology, idxs in topology_to_indices.items():
        if topology in train_topos:
            train_indices.extend(idxs)
        elif topology in val_topos:
            valid_indices.extend(idxs)
        elif topology in test_topos:
            test_indices.extend(idxs)
        elif topology in ood_topos:
            ood_indices.extend(idxs)

    return {
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "test_indices": test_indices,
        "ood_indices": ood_indices,
        "train_topos": train_topos,
        "val_topos": val_topos,
        "test_topos": test_topos,
        "ood_topos": ood_topos,
        "topology_to_indices": topology_to_indices,
    }


def build_dataloaders(
    pc_dataset,
    batch_size=16,
    num_workers=5,
    valid_size=0.1,
    test_size=0.1,
    ood_size=0.1,
    seed=42,
    K=None,
):
    splits = split_indices_by_topology(
        pc_dataset,
        seed=seed,
        ood_fraction=ood_size,
        valid_fraction=valid_size,
        test_fraction=test_size,
    )

    train_subset = Subset(pc_dataset, splits["train_indices"])
    valid_subset = Subset(pc_dataset, splits["valid_indices"])
    test_subset = Subset(pc_dataset, splits["test_indices"])
    ood_subset = Subset(pc_dataset, splits["ood_indices"])

    train_loader = data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda batch: create_batch(batch, K=K),
    )

    valid_loader = data.DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: create_batch(batch, K=K),
    )

    test_loader = data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: create_batch(batch, K=K),
    )

    ood_loader = data.DataLoader(
        ood_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: create_batch(batch, K=K),
    )

    return {
        **splits,
        "train_subset": train_subset,
        "valid_subset": valid_subset,
        "test_subset": test_subset,
        "ood_subset": ood_subset,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "ood_loader": ood_loader,
    }


def prepare_pointcloud_data(
    address,
    downsample_fraction=1 / 3,
    downsample_group_cols=("topology_id", "colony_id"),
    downsample_random_state=42,
    time_point=250,
    gene_cols=("g1", "g2", "g3"),
    group_cols=("topology_id", "colony_id"),
    batch_size=16,
    num_workers=5,
    valid_size=0.1,
    test_size=0.1,
    ood_size=0.1,
    seed=42,
    K=None,
):
    adata = load_adata(address)
    adata = downsample_adata_by_colony(
        adata,
        frac=downsample_fraction,
        group_cols=downsample_group_cols,
        random_state=downsample_random_state,
    )

    pc_dataset = ColonyFromAnnDataDataset(
        adata=adata,
        time_point=time_point,
        gene_cols=gene_cols,
        group_cols=group_cols,
    )
    pc_dataset_0 = ColonyFromAnnDataDataset(
        adata=adata,
        time_point=0,
        gene_cols=gene_cols,
        group_cols=group_cols,
    )

    loaders = build_dataloaders(
        pc_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        valid_size=valid_size,
        test_size=test_size,
        ood_size=ood_size,
        seed=seed,
        K=K,
    )

    return {
        "adata": adata,
        "pc_dataset": pc_dataset,
        "pc_dataset_0": pc_dataset_0,
        **loaders,
        "N_elements_in_set": pc_dataset[0]["seq_length"],
    }
