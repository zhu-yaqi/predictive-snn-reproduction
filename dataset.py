
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from spikingjelly.datasets.n_mnist import NMNIST


class NMNISTFrames(Dataset):
    """Convert N-MNIST frames into float tensors."""

    def __init__(self, root, train, T=16, scale="max"):
        self.scale = scale

        self.dataset = NMNIST(
            root=str(root),
            train=train,
            data_type="frame",
            frames_number=T,
            split_by="number"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames, label = self.dataset[idx]

        # [T, 2, 34, 34]
        x = torch.as_tensor(frames, dtype=torch.float32)

        # Normalize per sample; an experimental choice.
        if self.scale == "max":
            x = x / x.max().clamp(min=1.0)
        elif self.scale == "binary":
            x = (x > 0).float()
        elif self.scale != "none":
            raise ValueError(self.scale)

        # [T, 2312]
        x = x.flatten(start_dim=1)

        y = int(label)

        assert x.shape[1] == 2312

        return x, y


def make_loaders(
    root="./data/nmnist",
    T=16,
    batch_size=64,
    num_workers=0,
    scale="max",
    seed=42
):
    train_full = NMNISTFrames(
        root=root,
        train=True,
        T=T,
        scale=scale
    )

    test_set = NMNISTFrames(
        root=root,
        train=False,
        T=T,
        scale=scale
    )

    # Fixed 90/10 training/validation split
    n_val = int(len(train_full) * 0.1)
    n_train = len(train_full) - n_val

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(
        train_full,
        [n_train, n_val],
        generator=generator
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **common
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **common
    )

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **common
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="./data/nmnist"
    )
    args = parser.parse_args()

    train_loader, val_loader, test_loader = make_loaders(
        root=args.root
    )

    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    x, y = next(iter(train_loader))

    print("Input shape:", x.shape)
    print("Labels shape:", y.shape)
    print("Range:", x.min().item(), x.max().item())