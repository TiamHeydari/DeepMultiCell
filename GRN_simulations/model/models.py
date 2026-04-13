import torch
import torch.nn as nn
 
#====================================Moded v1==============================================


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


class DeepSet_Auto_encoder_v1(nn.Module):
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

        self.latent_layer_mu = nn.Linear(feats_encoder[-1], feats_latent)
        self.latent_layer_logvar = nn.Linear(feats_encoder[-1], feats_latent)

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

        batch_size = x_hat.shape[0]
        x_hat = x_hat.reshape(batch_size, self.N_elements_in_set, self.in_features)

        return x_hat, z, mu, logvar






#====================================Moded v2==============================================


import torch
import torch.nn as nn


class MAB(nn.Module):
    """
    Multihead Attention Block from Set Transformer style design.

    Q : (B, Nq, D)
    K : (B, Nk, D)

    Output:
        (B, Nq, D)
    """
    def __init__(self, dim_Q: int, dim_KV: int, dim_hidden: int, num_heads: int, ln: bool = False):
        super().__init__()

        self.fc_q = nn.Linear(dim_Q, dim_hidden)
        self.fc_k = nn.Linear(dim_KV, dim_hidden)
        self.fc_v = nn.Linear(dim_KV, dim_hidden)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=num_heads,
            batch_first=True,
        )

        self.fc_o = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        self.ln0 = nn.LayerNorm(dim_hidden) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_hidden) if ln else nn.Identity()

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q_proj = self.fc_q(Q)   # (B, Nq, D)
        K_proj = self.fc_k(K)   # (B, Nk, D)
        V_proj = self.fc_v(K)   # (B, Nk, D)

        H, _ = self.attn(Q_proj, K_proj, V_proj)   # (B, Nq, D)
        H = self.ln0(Q_proj + H)
        
        O = self.ln1(H + self.fc_o(H))
        return O


class SAB(nn.Module):
    """
    Self-Attention Block: MAB(X, X)
    """
    def __init__(self, dim_in: int, dim_hidden: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_hidden, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
    Uses a small number of learned seed vectors to pool a set.

    Here we will use num_seeds=1 so the whole colony becomes one vector.
    """
    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, ln: bool = False):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)   # (B, num_seeds, D)
        return self.mab(S, X)          # (B, num_seeds, D)


class SetTransformerEncoder(nn.Module):
    """
    Set Transformer encoder that maps a set (B, N, C) to one vector (B, D).
    """
    def __init__(
        self,
        in_features: int,
        dim_hidden: int,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        ln: bool = True,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_features, dim_hidden)

        sab_layers = []
        for _ in range(num_sab_layers):
            sab_layers.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.sab_layers = nn.ModuleList(sab_layers)

        # pool the whole set into ONE vector
        self.pma = PMA(dim_hidden, num_heads, num_seeds=1, ln=ln)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, C)
        returns:
            h_set : (B, D)
        """
        x = self.input_proj(x)  # (B, N, D)

        for layer in self.sab_layers:
            x = layer(x)        # (B, N, D)

        h = self.pma(x)         # (B, 1, D)
        h = h.squeeze(1)        # (B, D)
        return h



class DeepSet_Auto_encoder_v2_SetTransformer(nn.Module):
    """
    VAE-style set autoencoder:
    - Set Transformer encoder
    - one latent point per colony
    - same flat decoder as your current model

    This is a controlled experiment:
    only the encoder is upgraded.
    """
    def __init__(
        self,
        in_features: int,
        feats_encoder: list,
        feats_latent: int,
        feats_decoder: list,
        N_elements_in_set: int,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.feats_encoder = feats_encoder
        self.feats_latent = feats_latent
        self.feats_decoder = feats_decoder
        self.N_elements_in_set = N_elements_in_set

        if len(feats_encoder) == 0:
            raise ValueError("feats_encoder must contain at least one hidden dimension.")

        dim_hidden = feats_encoder[-1]

        # ---------------- Encoder ----------------
        self.encoder = SetTransformerEncoder(
            in_features=in_features,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_sab_layers=num_sab_layers,
            ln=use_layernorm,
        )

        # ---------------- Latent ----------------
        self.latent_layer_mu = nn.Linear(dim_hidden, feats_latent)
        self.latent_layer_logvar = nn.Linear(dim_hidden, feats_latent)

        # ---------------- Decoder ----------------
        decoder_layers = []
        decoder_layers.append(nn.Linear(feats_latent, feats_decoder[0]))
        decoder_layers.append(nn.ReLU())

        for i in range(1, len(feats_decoder)):
            decoder_layers.append(nn.Linear(feats_decoder[i - 1], feats_decoder[i]))
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.Linear(feats_decoder[-1], in_features * N_elements_in_set))
        self.decoder = nn.ModuleList(decoder_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # -------- encoder: set -> one vector --------
        h_set = self.encoder(x)                  # (B, D)

        # -------- latent --------
        mu = self.latent_layer_mu(h_set)         # (B, d_z)
        logvar = self.latent_layer_logvar(h_set) # (B, d_z)
        z = self.reparameterize(mu, logvar)      # (B, d_z)

        # -------- decoder --------
        x_hat = z
        for layer in self.decoder:
            x_hat = layer(x_hat)                 # (B, N*C)

        batch_size = x_hat.shape[0]
        x_hat = x_hat.reshape(batch_size, self.N_elements_in_set, self.in_features)

        return x_hat, z, mu, logvar




#====================================Moded v3==============================================


class SeedDecoder(nn.Module):
    """
    Seed / query decoder with optional ONE self-attention block.

    Base idea:
        [z, seed_i] -> shared decoder -> x_hat_i

    Optional variant:
        [z, seed_i] -> projection -> one self-attention block -> shared decoder -> x_hat_i
    """
    def __init__(
        self,
        feats_latent: int,
        feats_decoder: list,
        out_features: int,
        N_elements_in_set: int,
        seed_dim: int = 16,
        num_heads: int = 4,
        use_self_attn: bool = True,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if len(feats_decoder) == 0:
            raise ValueError("feats_decoder must contain at least one hidden dimension.")

        self.feats_latent = feats_latent
        self.feats_decoder = feats_decoder
        self.out_features = out_features
        self.N_elements_in_set = N_elements_in_set
        self.seed_dim = seed_dim
        self.num_heads = num_heads
        self.use_self_attn = use_self_attn

        # One learned seed per output slot
        self.decoder_seeds = nn.Parameter(
            torch.randn(1, N_elements_in_set, seed_dim)
        )  # shape: (1, N, seed_dim)

        decoder_input_dim = feats_latent + seed_dim
        decoder_dim = feats_decoder[0]

        # First projection from [z, seed] into decoder token space
        self.input_proj = nn.Sequential(
            nn.Linear(decoder_input_dim, decoder_dim),
            nn.ReLU(),
        )

        # Optional one self-attention block among output slots
        if use_self_attn:
            self.self_attn = SAB(
                dim_in=decoder_dim,
                dim_hidden=decoder_dim,
                num_heads=num_heads,
                ln=use_layernorm,
            )
        else:
            self.self_attn = nn.Identity()

        # Remaining pointwise decoder MLP
        layers = []
        in_dim = decoder_dim

        for hidden_dim in feats_decoder[1:]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_features))
        self.point_decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, d_z)

        returns
        -------
        x_hat : (B, N, out_features)
        """
        B = z.shape[0]

        z_expanded = z.unsqueeze(1).expand(-1, self.N_elements_in_set, -1)  # (B, N, d_z)
        seeds = self.decoder_seeds.expand(B, -1, -1)                        # (B, N, seed_dim)

        x = torch.cat([z_expanded, seeds], dim=-1)  # (B, N, d_z + seed_dim)

        # project into decoder token space
        x = self.input_proj(x)                      # (B, N, feats_decoder[0])

        # optional one self-attention block
        x = self.self_attn(x)                       # (B, N, feats_decoder[0])

        # pointwise decoding
        x_hat = self.point_decoder(x)              # (B, N, out_features)

        return x_hat


class DeepSet_Auto_encoder_v3_SetTransformer(nn.Module):
    """
    VAE-style set autoencoder:
    - Set Transformer encoder
    - one latent point per colony
    - decoupled SeedDecoder module
    """
    def __init__(
        self,
        in_features: int,
        feats_encoder: list,
        feats_latent: int,
        feats_decoder: list,
        N_elements_in_set: int,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        use_layernorm: bool = True,
        seed_dim: int = 16,
        decoder_self_attn: bool = True,
    ) -> None:
        super().__init__()

        if len(feats_encoder) == 0:
            raise ValueError("feats_encoder must contain at least one hidden dimension.")

        self.in_features = in_features
        self.feats_encoder = feats_encoder
        self.feats_latent = feats_latent
        self.feats_decoder = feats_decoder
        self.N_elements_in_set = N_elements_in_set
        self.seed_dim = seed_dim

        dim_hidden = feats_encoder[-1]

        # ---------------- Encoder ----------------
        self.encoder = SetTransformerEncoder(
            in_features=in_features,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_sab_layers=num_sab_layers,
            ln=use_layernorm,
        )

        # ---------------- Latent ----------------
        self.latent_layer_mu = nn.Linear(dim_hidden, feats_latent)
        self.latent_layer_logvar = nn.Linear(dim_hidden, feats_latent)

        # ---------------- Decoder ----------------
        self.decoder = SeedDecoder(
            feats_latent=feats_latent,
            feats_decoder=feats_decoder,
            out_features=in_features,
            N_elements_in_set=N_elements_in_set,
            seed_dim=seed_dim,
            num_heads=num_heads,
            use_self_attn=decoder_self_attn,
            use_layernorm=use_layernorm,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        h_set = self.encoder(x)                  # (B, D)

        mu = self.latent_layer_mu(h_set)         # (B, d_z)
        logvar = self.latent_layer_logvar(h_set) # (B, d_z)
        z = self.reparameterize(mu, logvar)      # (B, d_z)

        x_hat = self.decoder(z)                  # (B, N, in_features)

        return x_hat, z, mu, logvar


# ==================================== V4 ====================================

# this will have the same encoder as v3, but a more sophisticated decoder that uses one self-attention 
class AttentiveSeedDecoder(nn.Module):
    """
    Seed / query decoder with:
      1) cross-attention from seeds to the single latent token z
      2) one self-attention block among decoder slots
      3) pointwise MLP to output each reconstructed point

    Flow:
        learned seeds
            -> seed projection
            -> pointwise MLP block
            -> self-attention among slots
            -> pointwise MLP head
            -> x_hat
    """
    def __init__(
        self,
        feats_latent: int,
        feats_decoder: list,
        out_features: int,
        N_elements_in_set: int,
        seed_dim: int = 16,
        num_heads: int = 4,
        ln: bool = True,
    ) -> None:
        super().__init__()

        if len(feats_decoder) == 0:
            raise ValueError("feats_decoder must contain at least one hidden dimension.")

        self.feats_latent = feats_latent
        self.feats_decoder = feats_decoder
        self.out_features = out_features
        self.N_elements_in_set = N_elements_in_set
        self.seed_dim = seed_dim
        self.num_heads = num_heads

        # use first decoder width as decoder token dimension
        self.decoder_dim = feats_decoder[0]

        # learned seeds / slots
        self.decoder_seeds = nn.Parameter(
            torch.randn(1, N_elements_in_set, seed_dim)
        )  # (1, N, seed_dim)

        # project seeds and z into common decoder token space
        self.seed_proj = nn.Linear(seed_dim, self.decoder_dim)
        self.z_proj = nn.Linear(feats_latent, self.decoder_dim)

        # cross-attention: seeds query the single latent token z
        self.cross_attn = MAB(
            dim_Q=self.decoder_dim,
            dim_KV=self.decoder_dim,
            dim_hidden=self.decoder_dim,
            num_heads=num_heads,
            ln=ln,
        )

        # small pointwise block before self-attention
        self.pre_self_mlp = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.ReLU(),
        )

        # one self-attention block among output slots
        self.self_attn = SAB(
            dim_in=self.decoder_dim,
            dim_hidden=self.decoder_dim,
            num_heads=num_heads,
            ln=ln,
        )

        # pointwise head after self-attention
        post_layers = []
        in_dim = self.decoder_dim

        for hidden_dim in feats_decoder[1:]:
            post_layers.append(nn.Linear(in_dim, hidden_dim))
            post_layers.append(nn.ReLU())
            in_dim = hidden_dim

        post_layers.append(nn.Linear(in_dim, out_features))
        self.post_mlp = nn.Sequential(*post_layers)



    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, d_z)

        returns
        -------
        x_hat : (B, N, out_features)
        """
        B = z.shape[0]

        # learned seeds repeated across batch
        seeds = self.decoder_seeds.expand(B, -1, -1)   # (B, N, seed_dim)
        seed_tokens = self.seed_proj(seeds)            # (B, N, Dd)

        # single latent token
        z_token = self.z_proj(z).unsqueeze(1)          # (B, 1, Dd)

        # 1) cross-attention: seeds attend to z
        x = self.cross_attn(seed_tokens, z_token)      # (B, N, Dd)

        # small pointwise block
        x = self.pre_self_mlp(x)                       # (B, N, Dd)

        # 2) self-attention among slots
        x = self.self_attn(x)                          # (B, N, Dd)

        # 3) pointwise decoder head
        x_hat = self.post_mlp(x)                       # (B, N, out_features)

        return x_hat




class DeepSet_Auto_encoder_v4_SetTransformer(nn.Module):
    """
    VAE-style set autoencoder:
    - Set Transformer encoder
    - one latent point per colony
    - attentive seed decoder

    The colony is still compressed into ONE latent point z.
    Decoder improvements are only decoder-side scaffolding.
    """
    def __init__(
        self,
        in_features: int,
        feats_encoder: list,
        feats_latent: int,
        feats_decoder: list,
        N_elements_in_set: int,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        use_layernorm: bool = True,
        seed_dim: int = 16,
    ) -> None:
        super().__init__()

        if len(feats_encoder) == 0:
            raise ValueError("feats_encoder must contain at least one hidden dimension.")
        if len(feats_decoder) == 0:
            raise ValueError("feats_decoder must contain at least one hidden dimension.")

        self.in_features = in_features
        self.feats_encoder = feats_encoder
        self.feats_latent = feats_latent
        self.feats_decoder = feats_decoder
        self.N_elements_in_set = N_elements_in_set
        self.seed_dim = seed_dim

        dim_hidden = feats_encoder[-1]

        # ---------------- Encoder ----------------
        self.encoder = SetTransformerEncoder(
            in_features=in_features,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            num_sab_layers=num_sab_layers,
            ln=use_layernorm,
        )

        # ---------------- Latent ----------------
        self.latent_layer_mu = nn.Linear(dim_hidden, feats_latent)
        self.latent_layer_logvar = nn.Linear(dim_hidden, feats_latent)

        # ---------------- Decoder ----------------
        self.decoder = AttentiveSeedDecoder(
            feats_latent=feats_latent,
            feats_decoder=feats_decoder,
            out_features=in_features,
            N_elements_in_set=N_elements_in_set,
            seed_dim=seed_dim,
            num_heads=num_heads,
            ln=use_layernorm,
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # -------- encoder: set -> one vector --------
        h_set = self.encoder(x)                  # (B, D)

        # -------- latent --------
        mu = self.latent_layer_mu(h_set)         # (B, d_z)
        logvar = self.latent_layer_logvar(h_set) # (B, d_z)
        z = self.reparameterize(mu, logvar)      # (B, d_z)

        # -------- decoder --------
        x_hat = self.decoder(z)                  # (B, N, in_features)

        return x_hat, z, mu, logvar