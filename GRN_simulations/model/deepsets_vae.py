import torch
import torch.nn as nn


class DeepSet_Linear_Layer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalization: str = "",
        pool: str = "mean",
    ) -> None:
        super().__init__()

        self.Gamma = nn.Linear(in_features, out_features)
        self.Lambda = nn.Linear(in_features, out_features)

        self.normalization = normalization
        self.pool = pool

        if normalization == "batchnorm":
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, C)
        if self.pool == "mean":
            pooled = x.mean(dim=1, keepdim=True)
            x = self.Gamma(x) + self.Lambda(x - pooled)
        elif self.pool == "max":
            pooled = x.max(dim=1, keepdim=True).values
            x = self.Gamma(x) + self.Lambda(x - pooled)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool}")

        if self.normalization == "batchnorm":
            x = self.bn(x)

        return x


class DeepSet_VAE(nn.Module):
    def __init__(
        self,
        in_features: int,
        feats_encoder: list,
        feats_latent: int,
        feats_decoder: list,
        N_elements_in_set: int,
        normalization: str = "",
        pool: str = "mean",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.feats_encoder = feats_encoder
        self.feats_latent = feats_latent
        self.feats_decoder = feats_decoder
        self.N_elements_in_set = N_elements_in_set

        # Encoder
        layers = []
        layers.append(DeepSet_Linear_Layer(in_features, feats_encoder[0], normalization, pool))
        layers.append(nn.ReLU())

        for i in range(1, len(feats_encoder)):
            layers.append(
                DeepSet_Linear_Layer(
                    feats_encoder[i - 1], feats_encoder[i], normalization, pool
                )
            )
            layers.append(nn.ReLU())

        self.encoder = nn.ModuleList(layers)

        # Latent heads
        self.latent_layer_mu = nn.Linear(feats_encoder[-1], feats_latent)
        self.latent_layer_logvar = nn.Linear(feats_encoder[-1], feats_latent)

        # Decoder
        layers = []
        layers.append(nn.Linear(feats_latent, feats_decoder[0]))
        layers.append(nn.ReLU())

        for i in range(1, len(feats_decoder)):
            layers.append(nn.Linear(feats_decoder[i - 1], feats_decoder[i]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(feats_decoder[-1], in_features * N_elements_in_set))
        self.decoder = nn.ModuleList(layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        for layer in self.encoder:
            x = layer(x)

        x = x.mean(dim=1)

        mu = self.latent_layer_mu(x)
        logvar = self.latent_layer_logvar(x)
        z = self.reparameterize(mu, logvar)

        x_hat = z
        for layer in self.decoder:
            x_hat = layer(x_hat)

        B = x_hat.shape[0]
        x_hat = x_hat.reshape(B, self.N_elements_in_set, self.in_features)

        return x_hat, z, mu, logvar
