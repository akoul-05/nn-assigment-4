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
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from homework.models import load_model, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import PlannerMetric


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


def build_loaders(batch_size: int, num_workers: int, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_ds = RoadDataset(split="train")
    val_ds = RoadDataset(split="val")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader


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

    # Heuristic weight decay default (slightly lower for MLP)
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
