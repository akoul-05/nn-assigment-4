from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

# Used Copilot & Chatgpt to help implement the models below; most of the code was referenced.

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        # hidden_dims=(256, 256, 128),
        hidden_dims=(512, 512, 256),

    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        in_dim = n_track * 4 * 4

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, n_waypoints * 2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # b = track_left.size(0)
        # x = torch.cat([track_left, track_right], dim=1)   # (b, 2*n_track, 2)
        # x = x.reshape(b, -1)                              # (b, 4*n_track)
        # out = self.net(x)                                 # (b, n_waypoints*2)
        # return out.view(b, self.n_waypoints, 2)
    
        b = track_left.size(0)
        center = 0.5 * (track_left + track_right)
        width  = (track_right - track_left)
        x = torch.cat([track_left, track_right, center, width], dim=1)
        x = x.reshape(b, -1)
        out = self.net(x)
        return out.view(b, self.n_waypoints, 2)

        #raise NotImplementedError


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        #self.query_embed = nn.Embedding(n_waypoints, d_model)

        self.d_model = d_model

        self.point_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.side_embed = nn.Embedding(2, d_model)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project query output to (x,y)
        self.head = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.size(0)

        mem_xy = torch.cat([track_left, track_right], dim=1)     
        mem = self.point_encoder(mem_xy)

        side_ids = torch.cat(
            [
                torch.zeros((b, self.n_track), dtype=torch.long, device=mem.device),
                torch.ones((b, self.n_track), dtype=torch.long, device=mem.device),
            ],
            dim=1,
        )
        mem = mem + self.side_embed(side_ids)

        # Queries
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        # Cross-attention
        dec = self.decoder(tgt=q, memory=mem)
        out = self.head(dec)
        return out
        #raise NotImplementedError


class PatchEmbedding(nn.Module):
    def __init__(self, h: int = 96, w: int = 128, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 64):
        """
        Convert image to sequence of patch embeddings using a simple approach

        This is provided as a helper for implementing the Vision Transformer Planner.
        You can use this directly in your ViTPlanner implementation.

        Args:
            h: height of input image
            w: width of input image
            patch_size: size of each patch
            in_channels: number of input channels (3 for RGB)
            embed_dim: embedding dimension
        """
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p) -> (B, C, H//p, W//p, p, p)
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 1, 2, 4, 3, 5)
        # Flatten patches: (B, C, H//p, W//p, p*p) -> (B, H//p * W//p, C * p * p)
        num_patches = (H // p) * (W // p)
        x = x.reshape(B, num_patches, C * p * p)

        # Linear projection
        return self.projection(x)
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        A single Transformer encoder block with multi-head attention and MLP.

        You can use the one you implemented in Homework 3.

        Hint: A transformer block typically consists of:
        1. Layer normalization
        2. Multi-head self-attention (use torch.nn.MultiheadAttention with batch_first=True)
        3. Residual connection
        4. Layer normalization
        5. MLP (Linear -> GELU -> Dropout -> Linear -> Dropout)
        6. Residual connection

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dimension to embedding dimension
            dropout: dropout probability
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        hidden = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

        #raise NotImplementedError("TransformerBlock.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, embed_dim) input sequence

        Returns:
            (batch_size, sequence_length, embed_dim) output sequence
        """
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(attn_out)

        # Feedforward
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x
    #raise NotImplementedError("TransformerBlock.forward() is not implemented")


class ViTPlanner(torch.nn.Module):
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
        """
        Vision Transformer (ViT) based planner that predicts waypoints from images.

        Args:
            n_waypoints (int): number of waypoints to predict

        Hint - you can add more arguments to the constructor such as:
            patch_size: int, size of image patches
            embed_dim: int, embedding dimension
            num_layers: int, number of transformer layers
            num_heads: int, number of attention heads

        Note: You can use the provided PatchEmbedding and TransformerBlock classes.
        The input images are of size (96, 128).

        Hint: A typical ViT architecture consists of:
        1. Patch embedding layer to convert image into sequence of patches
        2. Positional embeddings (learnable parameters) added to patch embeddings
        3. Multiple transformer encoder blocks
        4. Final normalization layer
        5. Output projection to predict waypoints

        Hint: For this task, you can either:
        - Use a classification token ([CLS]) approach like in standard ViT as global image representation
        - Use learned query embeddings (similar to TransformerPlanner)
        - Average pool over all patch features
        """
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            h=img_h, w=img_w, patch_size=patch_size, in_channels=3, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        blocks = []
        for _ in range(num_layers):
            blocks.append(
                TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            )
        self.blocks = nn.Sequential(*blocks)
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


        # self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        # self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        #raise NotImplementedError("ViTPlanner.__init__() is not implemented")

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, 96, 128) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n_waypoints, 2)

        Hint: The typical forward pass consists of:
        1. Normalize input image
        2. Convert image to patch embeddings
        3. Add positional embeddings
        4. Pass through transformer blocks
        5. Extract features for prediction (e.g., [CLS] token or average pooling)
        6. Project to waypoint coordinates
        """
        #x = image
        #x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # 1) Normalize
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # 2) Patch
        x = self.patch_embed(x)

        # 3) Add positional encodings
        x = x + self.pos[:, : x.size(1), :]

        # 4) Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # 5) Global average pool over tokens
        x = x.mean(dim=1)

        # 6) Project to waypoints
        out = self.head(x)  # (B, n_waypoints*2)
        return out.view(x.size(0), self.n_waypoints, 2)

        #raise NotImplementedError("ViTPlanner.forward() is not implemented")


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "vit_planner": ViTPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
