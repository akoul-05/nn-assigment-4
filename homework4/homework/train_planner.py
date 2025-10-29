"""
Usage:
  # MLP
  python3 -m homework.train_planner --model mlp_planner --epochs 50 --lr 1e-3 --batch_size 128
  # Transformer
  python3 -m homework.train_planner --model transformer_planner --epochs 60 --lr 3e-4 --batch_size 128
  # ViT
  python3 -m homework.train_planner --model vit_planner --epochs 80 --lr 3e-4 --batch_size 64
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from homework.models import load_model, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import PlannerMetric


# ---------------------------
# Arg parsing / seeding
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   choices=["mlp_planner", "transformer_planner", "vit_planner"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Dataset discovery utilities
# ---------------------------

EP_KEY_FILE = "info.npz"  # file that marks an episode directory (e.g., drive_data/train/*/info.npz)


def _possible_roots() -> list[Path]:
    """Try a few likely places for drive_data/ so cwd quirks don't break things."""
    here = Path.cwd()
    return [
        here / "drive_data",
        here.parent / "drive_data",
        Path(__file__).resolve().parent.parent / "drive_data",  # project root (../ from homework/)
    ]


def _find_drive_data_root() -> Path:
    for cand in _possible_roots():
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "drive_data/ not found. cd to your project root (where bundle.py is) and unzip the dataset there."
    )


def _collect_episode_dirs(split_dir: Path) -> list[Path]:
    """
    Find episode directories by locating */info.npz and returning their parents.
    """
    info_files = list(split_dir.rglob(EP_KEY_FILE))
    ep_dirs = sorted({p.parent for p in info_files if p.is_file()})
    if ep_dirs:
        return ep_dirs

    # Fallback: any subdir that contains the key file
    candidates = []
    for d in sorted([p for p in split_dir.rglob("*") if p.is_dir()]):
        if (d / EP_KEY_FILE).exists():
            candidates.append(d)
    return candidates


def _try_instantiate_episode(ep_dir: Path):
    """
    Try common constructor signatures using the EPISODE DIRECTORY.
    Returns a RoadDataset or None.
    """
    # Most common: episode_path=<dir>
    try:
        return RoadDataset(episode_path=str(ep_dir))
    except TypeError:
        pass
    except Exception:
        return None

    # Other names some repos use
    for kw in ("path", "dir", "episode_dir"):
        try:
            return RoadDataset(**{kw: str(ep_dir)})
        except Exception:
            continue
    return None


def _build_split_dataset(split_dir: Path) -> ConcatDataset:
    ep_dirs = _collect_episode_dirs(split_dir)
    if not ep_dirs:
        raise RuntimeError(
            f"No episode directories found under {split_dir} (looked for */{EP_KEY_FILE})."
        )

    datasets = []
    for d in ep_dirs:
        ds = _try_instantiate_episode(d)
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        sample = "\n".join(str(p) for p in ep_dirs[:10])
        raise RuntimeError(
            f"Could not instantiate RoadDataset for any episode dir under {split_dir}.\n"
            f"Tried {len(ep_dirs)} dirs, e.g.:\n{sample}\n"
            "Open homework/datasets/road_dataset.py and confirm the parameter name. "
            "Supported keys here: episode_path / path / dir / episode_dir."
        )

    return ConcatDataset(datasets)


def build_loaders(batch_size: int, num_workers: int, device: torch.device):
    """
    Build train/val loaders by scanning:
        drive_data/train/*/info.npz  and  drive_data/val/*/info.npz
    and instantiating one RoadDataset per episode directory. If no explicit split
    exists, build from the whole root and perform a 90/10 random split.
    """
    pin = device.type == "cuda"
    root = _find_drive_data_root()

    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.exists() and val_dir.exists():
        print(f"[build_loaders] Using split dirs: {train_dir} | {val_dir}")
        train_ds = _build_split_dataset(train_dir)
        val_ds = _build_split_dataset(val_dir)
    else:
        # No explicit split: attempt to build from the whole root, then random split.
        print(f"[build_loaders] No train/val dirs; building from {root} and random-splitting.")
        full_ds = _build_split_dataset(root)
        n = len(full_ds)
        n_train = max(1, int(0.9 * n))
        n_val = max(1, n - n_train)
        gen = torch.Generator().manual_seed(1337)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    print(f"[build_loaders] train_samples≈{len(train_ds)} | val_samples≈{len(val_ds)}")
    return train_loader, val_loader


# ---------------------------
# Training / Eval
# ---------------------------

def forward_model(model: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    name = model.__class__.__name__.lower()
    if "vit" in name:
        return model(image=batch["image"].to(device))
    else:
        return model(
            track_left=batch["track_left"].to(device),
            track_right=batch["track_right"].to(device),
        )


def compute_loss(pred: torch.Tensor, batch: dict, device: torch.device) -> torch.Tensor:
    target = batch["waypoints"].to(device)                  # (B, 3, 2)
    mask = batch.get("waypoints_mask", None)                # (B, 3) or None
    loss = F.l1_loss(pred, target, reduction="none")        # (B, 3, 2)
    if mask is not None:
        m = mask.to(device)[..., None].float()              # (B, 3, 1)
        loss = (loss * m).sum() / m.sum().clamp_min(1.0)
    else:
        loss = loss.mean()
    return loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device):
    model.eval()
    tot_loss = 0.0
    seen = 0

    metric = PlannerMetric()

    for batch in val_loader:
        pred = forward_model(model, batch, device)          # (B, 3, 2)
        loss = compute_loss(pred, batch, device)
        bsz = pred.size(0)
        tot_loss += loss.item() * bsz
        seen += bsz

        labels = batch["waypoints"]
        labels_mask = batch.get("waypoints_mask", torch.ones(labels.shape[:2], dtype=torch.bool))
        metric.add(pred.cpu(), labels.cpu(), labels_mask.cpu())

    avg_loss = tot_loss / max(1, seen)
    m = metric.compute()
    return avg_loss, m["longitudinal_error"], m["lateral_error"], m["l1_error"]


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Training {args.model} on {device} for {args.epochs} epochs")

    # Heuristic: slightly lower default WD for MLP unless overridden
    wd = 0.01 if (args.model == "mlp_planner" and args.weight_decay == 0.05) else args.weight_decay

    model = load_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)

    train_loader, val_loader = build_loaders(args.batch_size, args.num_workers, device)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)

            pred = forward_model(model, batch, device)      # (B, 3, 2)
            loss = compute_loss(pred, batch, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = pred.size(0)
            running += loss.item() * bsz
            seen += bsz

            if step % 50 == 0:
                print(f"[Epoch {epoch} | {step}/{len(train_loader)}] train_loss={running/seen:.4f}")

        val_loss, long_err, lat_err, l1_err = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | long={long_err:.3f} | lat={lat_err:.3f} | L1={l1_err:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            path = save_model(model)
            print(f"✓ Saved best model to: {path}")

    # Final save just in case
    final_path = save_model(model)
    print(f"Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
