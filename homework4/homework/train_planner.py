# homework/train_planner.py

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split

from homework.models import load_model  # keeps grader compatibility
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import PlannerMetric


# ---------------------------
# CLI / Utilities
# ---------------------------

def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plan-and-train runners for MLP / Transformer / ViT.")
    ap.add_argument("--model", required=True,
                    choices=["mlp_planner", "transformer_planner", "vit_planner"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save_metric", default="lateral",
                    choices=["val_loss", "lateral", "longitudinal", "l1"])
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=0)  # 0 = no early stop
    return ap.parse_args()


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# Data discovery & loading
# ---------------------------

_EP_KEY = "info.npz"  # marks an episode directory

def _candidate_roots() -> list[Path]:
    here = Path.cwd()
    return [
        here / "drive_data",
        here.parent / "drive_data",
        Path(__file__).resolve().parent.parent / "drive_data",  # project root
    ]


def resolve_drive_data_root() -> Path:
    for root in _candidate_roots():
        if root.exists():
            return root
    raise FileNotFoundError(
        "drive_data/ not found. cd to the project root (where bundle.py lives) and unzip there."
    )


def _episode_dirs(under: Path) -> list[Path]:
    # Prefer exact key; fall back to any dir that contains the key file.
    infos = list(under.rglob(_EP_KEY))
    eps = sorted({p.parent for p in infos})
    if eps:
        return eps
    out = []
    for d in sorted(p for p in under.rglob("*") if p.is_dir()):
        if (d / _EP_KEY).exists():
            out.append(d)
    return out


def _try_build_episode(ep_dir: Path):
    # Some templates change the RoadDataset kwarg; try a few.
    try:
        return RoadDataset(episode_path=str(ep_dir))
    except TypeError:
        pass
    except Exception:
        return None

    for key in ("path", "dir", "episode_dir"):
        try:
            return RoadDataset(**{key: str(ep_dir)})
        except Exception:
            continue
    return None


def concat_split(split_root: Path) -> ConcatDataset:
    ep_dirs = _episode_dirs(split_root)
    if not ep_dirs:
        raise RuntimeError(f"No episodes found under {split_root} (looked for */{_EP_KEY}).")

    parts = []
    for e in ep_dirs:
        ds = _try_build_episode(e)
        if ds is not None:
            parts.append(ds)

    if not parts:
        examples = "\n".join(map(str, ep_dirs[:10]))
        raise RuntimeError(
            f"Could not instantiate RoadDataset for any episode under {split_root}.\n"
            f"Tried {len(ep_dirs)} dirs, e.g.:\n{examples}\n"
            "Check homework/datasets/road_dataset.py constructor arguments."
        )
    return ConcatDataset(parts)


def make_dataloaders(batch_size: int, num_workers: int, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    pin = (device.type == "cuda")
    root = resolve_drive_data_root()

    train_root = root / "train"
    val_root = root / "val"

    if train_root.exists() and val_root.exists():
        print(f"[data] Using pre-split dirs: {train_root} | {val_root}")
        train_ds = concat_split(train_root)
        val_ds = concat_split(val_root)
    else:
        print(f"[data] No explicit splits found; random-splitting from {root}.")
        full = concat_split(root)
        n_total = len(full)
        n_train = max(1, int(0.9 * n_total))
        n_val = max(1, n_total - n_train)
        gen = torch.Generator().manual_seed(1337)
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    print(f"[data] train≈{len(train_ds)} | val≈{len(val_ds)}")
    return train_loader, val_loader


# ---------------------------
# Core training/eval logic
# ---------------------------

def model_forward(m: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    name = m.__class__.__name__.lower()
    if "vit" in name:
        return m(image=batch["image"].to(device))
    return m(
        track_left=batch["track_left"].to(device),
        track_right=batch["track_right"].to(device),
    )


def loss_fn(pred: torch.Tensor, batch: dict, device: torch.device) -> torch.Tensor:
    target = batch["waypoints"].to(device)
    mask = batch.get("waypoints_mask", None)

    # Smooth L1 with a slightly higher weight on lateral (y) error
    comp = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none")  # (B, 3, 2)
    weights = torch.tensor([1.0, 1.8], device=device)  # [longitudinal, lateral]
    comp = comp * weights.view(1, 1, 2)

    if mask is not None:
        m = mask.to(device)[..., None].float()
        return (comp * m).sum() / m.sum().clamp_min(1.0)
    return comp.mean()


@torch.no_grad()
def validate(m: torch.nn.Module, loader: DataLoader, device: torch.device):
    m.eval()
    total, count = 0.0, 0
    metric = PlannerMetric()

    for batch in loader:
        pred = model_forward(m, batch, device)
        loss = loss_fn(pred, batch, device)

        bsz = pred.size(0)
        total += loss.item() * bsz
        count += bsz

        labels = batch["waypoints"]
        mask = batch.get("waypoints_mask", torch.ones(labels.shape[:2], dtype=torch.bool))
        metric.add(pred.cpu(), labels.cpu(), mask.cpu())

    avg_loss = total / max(1, count)
    mvals = metric.compute()
    return avg_loss, mvals["longitudinal_error"], mvals["lateral_error"], mvals["l1_error"]


def pick_score(val_loss: float, lon: float, lat: float, l1: float, key: str) -> float:
    if key == "val_loss":     return val_loss
    if key == "longitudinal": return lon
    if key == "lateral":      return lat
    if key == "l1":           return l1
    raise ValueError(f"Unknown metric: {key}")


# ---------------------------
# Runner
# ---------------------------

def main():
    import shutil

    args = parse_cli()
    set_all_seeds(args.seed)

    device = torch.device(args.device)
    print(f"[run] model={args.model} device={device} epochs={args.epochs}")

    # Mild WD tweak for MLP unless user overrides
    wd = 0.01 if (args.model == "mlp_planner" and args.weight_decay == 0.05) else args.weight_decay

    model = load_model(args.model).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    train_loader, val_loader = make_dataloaders(args.batch_size, args.num_workers, device)

    ckpt_dir = Path(__file__).resolve().parent  # .../homework
    best_file = ckpt_dir / f"{args.model}_best.th"
    last_file = ckpt_dir / f"{args.model}_last.th"
    required  = ckpt_dir / f"{args.model}.th"   # grader expects this exact name

    best_val = float("inf")
    best_epoch = -1
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0

        for step, batch in enumerate(train_loader, 1):
            optim.zero_grad(set_to_none=True)
            pred = model_forward(model, batch, device)
            loss = loss_fn(pred, batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            bsz = pred.size(0)
            running += loss.item() * bsz
            seen += bsz
            if step % 50 == 0:
                print(f"[epoch {epoch} | {step}/{len(train_loader)}] train_loss={running/max(1,seen):.4f}")

        v_loss, v_lon, v_lat, v_l1 = validate(model, val_loader, device)
        print(f"epoch {epoch:03d} | val_loss={v_loss:.4f} | long={v_lon:.3f} | lat={v_lat:.3f} | L1={v_l1:.3f}")

        # always save the last
        torch.save(model.state_dict(), last_file)

        score = pick_score(v_loss, v_lon, v_lat, v_l1, args.save_metric)
        if score < best_val - args.min_delta:
            best_val = score
            best_epoch = epoch
            torch.save(model.state_dict(), best_file)
            print(f"  ↳ new BEST ({args.save_metric}={best_val:.4f}) @ epoch {epoch}")
            stale = 0
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"[early-stop] no improvement for {args.patience} epochs.")
                break

    # ensure the grader-visible filename points at the best if available, else last
    if best_file.exists():
        shutil.copyfile(best_file, required)
        print(f"[save] BEST → {required.name}")
    else:
        shutil.copyfile(last_file, required)
        print(f"[save] LAST → {required.name}")

    if best_epoch > 0:
        print(f"[done] best {args.save_metric}={best_val:.4f} at epoch {best_epoch}")
    else:
        print("[done] no best checkpoint; used last weights")


if __name__ == "__main__":
    main()