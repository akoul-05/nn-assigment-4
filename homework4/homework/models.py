from pathlib import Path
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD  = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    """
    Predicts n_waypoints 2D points from track boundaries using a small MLP.
    Kept intentionally simple: only uses raw left/right boundaries.
    """
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        in_dim = n_track * 2 * 2

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **_) -> torch.Tensor:
        b = track_left.size(0)
        x = torch.cat([track_left, track_right], dim=1)
        x = x.reshape(b, -1)
        out = self.net(x)
        return out.view(b, self.n_waypoints, 2)

class TransformerPlanner(nn.Module):
    """
    Decoder-only transformer that attends learned waypoint queries over a memory
    built from the concatenated left/right points. No sinusoidal PE; we use a
    tiny feature vector [x, y, side] and a learned index embedding.
    """
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        in_feat = 3
        self.point_proj = nn.Sequential(
            nn.Linear(in_feat, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_embed = nn.Embedding(2 * n_track, d_model)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.mem_norm = nn.LayerNorm(d_model)
        self.q_norm   = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **_) -> torch.Tensor:
        """
        Inputs:
          track_left, track_right: (B, n_track, 2)
        Returns:
          (B, n_waypoints, 2)
        """
        B, T, _ = track_left.shape
        device = track_left.device
        mem_xy   = torch.cat([track_left, track_right], dim=1)
        side_l   = torch.zeros(B, T, 1, device=device)
        side_r   = torch.ones(B,  T, 1, device=device)
        mem_side = torch.cat([side_l, side_r], dim=1)
        mem_feat = torch.cat([mem_xy, mem_side], dim=-1)
        mem      = self.point_proj(mem_feat)
        idx = torch.arange(2 * T, device=device).unsqueeze(0).expand(B, -1)
        mem = mem + self.pos_embed(idx)
        mem = self.mem_norm(mem)

        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        q = self.q_norm(q)

        dec = self.decoder(tgt=q, memory=mem)
        out = self.head(dec)
        return out

class PatchEmbedding(nn.Module):
    """
    Simple patchification via Conv2d (kernel=stride=patch_size), then flatten to tokens.
    """
    def __init__(self, h: int = 96, w: int = 128, patch_size: int = 8,
                 in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        assert h % patch_size == 0 and w % patch_size == 0
        self.num_patches = (h // patch_size) * (w // patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Minimal encoder block: LN -> MHA -> Drop -> Residual -> LN -> MLP -> Residual
    """
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        hidden = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class ViTPlanner(nn.Module):
    """
    Tiny ViT that predicts n_waypoints 2D points from the image.
    Uses learned positional embeddings and global average pooling.
    """
    def __init__(
        self,
        n_waypoints: int = 3,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        img_h: int = 96,
        img_w: int = 128,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std",  torch.as_tensor(INPUT_STD),  persistent=False)

        self.patch_embed = PatchEmbedding(img_h, img_w, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, n_waypoints * 2),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor, **_) -> torch.Tensor:
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.patch_embed(x)
        x = x + self.pos[:, : x.size(1), :]
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        out = self.head(x)
        return out.view(x.size(0), self.n_waypoints, 2)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "vit_planner": ViTPlanner,
}

def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, ensure default model args match the saved weights."
            ) from e

    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m

def save_model(model: nn.Module) -> str:
    model_name = None
    for n, cls in MODEL_FACTORY.items():
        if type(model) is cls:
            model_name = n
            break
    if model_name is None:
        raise ValueError(f"Unsupported model type: {type(model)}")
    out_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), out_path)
    return out_path

def calculate_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024