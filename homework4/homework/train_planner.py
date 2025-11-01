# homework/trainer.py
# Train MLP / Transformer / ViT planners and save the LAST checkpoint only.

from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from homework.models import load_model
from homework.datasets.road_dataset import RoadDataset

EP_KEY = "info.npz"  # episode marker


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser("Simple planner trainer (save LAST only)")
    ap.add_argument("--model", required=True,
                    choices=["mlp_planner", "transformer_planner", "vit_planner"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Data
# ---------------------------
def _episodes(root: Path):
    return sorted({p.parent for p in root.rglob(EP_KEY)})

def _concat_from(split_root: Path) -> ConcatDataset:
    eps = _episodes(split_root)
    if not eps:
        raise RuntimeError(f"No episodes found under {split_root} (looked for */{EP_KEY}).")
    parts = [RoadDataset(episode_path=str(d)) for d in eps]
    return ConcatDataset(parts)

def make_loaders(batch_size: int, num_workers: int, device: torch.device):
    pin = device.type == "cuda"
    root = Path.cwd() / "drive_data"

    train_dir = root / "train"
    val_dir   = root / "val"

    if train_dir.exists() and val_dir.exists():
        print(f"[data] using splits: {train_dir} | {val_dir}")
        train_ds = _concat_from(train_dir)
        val_ds   = _concat_from(val_dir)
    else:
        print(f"[data] no explicit splits; random-splitting from {root}")
        full = _concat_from(root)
        n = len(full)
        n_train = max(1, int(0.9 * n))
        n_val   = max(1, n - n_train)
        gen = torch.Generator().manual_seed(1337)
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    print(f"[data] train≈{len(train_ds)} | val≈{len(val_ds)}")
    return train_loader, val_loader


# ---------------------------
# Train helpers
# ---------------------------
def forward_model(model: torch.nn.Module, batch: dict, device: torch.device):
    name = model.__class__.__name__.lower()
    if "vit" in name:
        return model(image=batch["image"].to(device))
    return model(track_left=batch["track_left"].to(device),
                 track_right=batch["track_right"].to(device))

def loss_fn(pred: torch.Tensor, batch: dict, device: torch.device):
    target = batch["waypoints"].to(device)
    mask = batch.get("waypoints_mask", None)

    # simple Smooth L1; you can weight lateral more if you want
    loss = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none")  # (B, 3, 2)
    if mask is not None:
        m = mask.to(device)[..., None].float()
        return (loss * m).sum() / m.sum().clamp_min(1.0)
    return loss.mean()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[run] model={args.model} epochs={args.epochs} device={device}")

    model = load_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader, val_loader = make_loaders(args.batch_size, args.num_workers, device)

    # where the grader expects weights
    out_path = Path(__file__).resolve().parent / f"{args.model}.th"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            opt.zero_grad(set_to_none=True)
            pred = forward_model(model, batch, device)
            loss = loss_fn(pred, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = pred.size(0)
            running += loss.item() * bsz
            seen += bsz
            if step % 50 == 0:
                print(f"[{epoch} | {step}/{len(train_loader)}] train_loss={running/max(1,seen):.4f}")

        # (optional) quick val loss for visibility; we still save LAST only
        with torch.no_grad():
            model.eval()
            tot, cnt = 0.0, 0
            for batch in val_loader:
                pred = forward_model(model, batch, device)
                loss = loss_fn(pred, batch, device)
                bsz = pred.size(0)
                tot += loss.item() * bsz
                cnt += bsz
            print(f"[{epoch}] val_loss={tot/max(1,cnt):.4f}")

        # save LAST every epoch (overwrites)
        torch.save(model.state_dict(), out_path)
        print(f"[save] wrote LAST → {out_path.name}")

    print("[done] training complete; LAST weights saved.")


if __name__ == "__main__":
    main()