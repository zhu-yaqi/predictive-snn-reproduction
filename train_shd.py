
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from spikingjelly.activation_based import functional

from shd_dataset import make_loaders
from baseline import BaselineSNN
from model import PredictiveSNN, compute_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_batch(frames, labels, device):
    """
    DataLoader: [B, T, D]
    Model:      [T, B, D]
    """
    x = frames.permute(1, 0, 2).contiguous()
    x = x.to(device, non_blocking=True)

    y = labels.long().to(
        device, non_blocking=True
    )

    return x, y


def get_loss(model_name, logits, labels, records):
    if model_name == "baseline":
        loss = F.cross_entropy(logits, labels)
        details = {
            "task": loss.detach(),
            "pred": loss.new_zeros(()),
            "rate": loss.new_zeros(())
        }
    else:
        loss, details = compute_loss(
            logits, labels, records
        )

    return loss, details


def communication_stats(records):
    """
    Per-sample communication activity proxy.

    Baseline: binary event count.
    Predictive suppression: sum of gated amplitudes.

    Both are summed over time, feature dimensions and layers.
    """
    batch_size = records[0]["y"].shape[1]

    comm_total = 0.0
    repr_total = 0.0

    for record in records:
        y = record["y"].detach()
        z = record["z"].detach()

        comm_total += y.abs().sum().item()
        repr_total += z.sum().item()

    return (
        comm_total / batch_size,
        repr_total / batch_size
    )


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    model_name,
    training
):
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    total_comm = 0.0
    total_repr = 0.0

    iterator = tqdm(
        loader,
        desc="Train" if training else "Evaluate",
        leave=False
    )

    for frames, labels in iterator:
        x, labels = prepare_batch(
            frames, labels, device
        )

        # Reset state at the start of an independent batch.
        functional.reset_net(model)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits, records = model(x)

            loss, details = get_loss(
                model_name,
                logits,
                labels,
                records
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        batch_size = labels.size(0)

        predictions = logits.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        total_loss += loss.item() * batch_size

        comm, repr_activity = communication_stats(
            records
        )

        total_comm += comm * batch_size
        total_repr += repr_activity * batch_size

        # Drop the previous batch's membrane state.
        functional.reset_net(model)

        iterator.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{total_correct / total_samples:.3f}"
        })

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "comm": total_comm / total_samples,
        "repr": total_repr / total_samples
    }


def save_checkpoint(
    path, model, optimizer, epoch,
    best_val_acc, args
):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "config": vars(args)
    }, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["baseline", "predictive"],
        required=True
    )

    parser.add_argument(
        "--data-root",
        default="./data/shd"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--scale",
        choices=["max", "binary", "none"],
        default="max"
    )
    parser.add_argument(
        "--output",
        default="./runs/shd"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.pt"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

#    print("Device:", device)
    import os

    print("=" * 50)
    print("Device:", device)

    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    )

    if torch.cuda.is_available():
        print(
            "PyTorch GPU:",
            torch.cuda.current_device()
        )
        print(
            "GPU name:",
            torch.cuda.get_device_name(
                torch.cuda.current_device()
            )
        )
        print(
            "Total GPU memory (GB):",
            round(
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).total_memory / 1024**3,
                2
            )
        )

    print("=" * 50)



    # Data
    train_loader, val_loader, test_loader = make_loaders(
        root=args.data_root,
        T=16,
        batch_size=args.batch_size,
        num_workers=args.workers,
        scale=args.scale,
        seed=args.seed
    )

    # Model
    if args.model == "baseline":
        model = BaselineSNN(
            in_dim=700,
            num_classes=20
        )
    else:
        model = PredictiveSNN(
            in_dim=700,
            num_classes=20
        )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    experiment = (
        f"{args.model}_seed{args.seed}"
    )

    out_dir = Path(args.output) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(
        out_dir / "config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False
        )

    log_path = out_dir / "history.csv"

    best_val_acc = -1.0
    best_epoch = -1
    start_epoch = 1

    if args.resume:
        checkpoint_path = out_dir / "last.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        # Validate checkpoint configuration.
        saved_config = checkpoint.get("config", {})

        for key in [
            "model",
            "seed",
            "scale",
            "lr",
            "batch_size",
            "data_root"
        ]:
            if key in saved_config:
                if getattr(args, key) != saved_config[key]:
                    raise ValueError(
                        f"Config mismatch: {key}, "
                        f"saved={saved_config[key]}, "
                        f"current={getattr(args, key)}"
                    )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        start_epoch = int(checkpoint["epoch"]) + 1

        # The older last.pt was saved before best_val_acc
        # was updated. Recover the actual best from best.pt.
        best_checkpoint_path = out_dir / "best.pt"

        if best_checkpoint_path.exists():
            best_checkpoint = torch.load(
                best_checkpoint_path,
                map_location="cpu",
                weights_only=True
            )
            best_val_acc = float(
                best_checkpoint["best_val_acc"]
            )
            best_epoch = int(
                best_checkpoint["epoch"]
            )
            del best_checkpoint
        else:
            best_val_acc = float(
                checkpoint["best_val_acc"]
            )

        print(
            f"Resuming from epoch {start_epoch}"
        )
        print(
            f"Best val acc: {best_val_acc:.4f}"
        )
    if args.resume and not log_path.exists():
        raise FileNotFoundError(
            f"Cannot resume without history: {log_path}"
        )

    log_mode = "a" if args.resume else "w"

    with open(
        log_path,
        log_mode,
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "train_comm",
                "val_comm",
                "val_repr"
            ]
        )
        if not args.resume:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            print(
                f"\nEpoch {epoch}/{args.epochs}"
            )

            train_stats = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                model_name=args.model,
                training=True
            )

            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                model_name=args.model,
                training=False
            )

            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["acc"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "train_comm": train_stats["comm"],
                "val_comm": val_stats["comm"],
                "val_repr": val_stats["repr"]
            }

            writer.writerow(row)
            f.flush()

            print(
                f"Train acc: {train_stats['acc']:.4f} | "
                f"Val acc: {val_stats['acc']:.4f}"
            )

            print(
                f"Val communication: "
                f"{val_stats['comm']:.2f}"
            )

            if val_stats["acc"] > best_val_acc:
                best_val_acc = val_stats["acc"]
                best_epoch = epoch

                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    best_val_acc,
                    args
                )

                print("Saved new best model.")

            save_checkpoint(
                out_dir / "last.pt",
                model,
                optimizer,
                epoch,
                best_val_acc,
                args
            )

    # Reload best checkpoint before final test.
    checkpoint = torch.load(
        out_dir / "best.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print("\nFinal test evaluation")

    test_stats = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        model_name=args.model,
        training=False
    )

    result = {
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test": test_stats
    }

    with open(
        out_dir / "result.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=2)

    print("\nResult:", result)


if __name__ == "__main__":
    main()