
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class SHDDataset(Dataset):
    def __init__(self, path, T=16, duration=1.4, scale="max"):
        self.path = str(Path(path).resolve())
        self.T = T
        self.duration = duration
        self.scale = scale
        self.file = None

        with h5py.File(self.path, "r") as f:
            self.labels = f["labels"][:].astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Each worker opens its own HDF5 handle.
        if self.file is None:
            self.file = h5py.File(self.path, "r")

        times = np.asarray(
            self.file["spikes/times"][idx],
            dtype=np.float64
        )
        units = np.asarray(
            self.file["spikes/units"][idx],
            dtype=np.int64
        )

        frames = np.zeros((self.T, 700), dtype=np.float32)

        valid = (
            (times >= 0) &
            (times <= self.duration) &
            (units >= 0) &
            (units < 700)
        )

        times = times[valid]
        units = units[valid]

        bins = np.floor(
            times / self.duration * self.T
        ).astype(np.int64)
        bins = np.clip(bins, 0, self.T - 1)

        np.add.at(frames, (bins, units), 1)

        if self.scale == "max":
            maximum = frames.max()
            if maximum > 0:
                frames /= maximum
        elif self.scale == "binary":
            frames = (frames > 0).astype(np.float32)
        elif self.scale != "none":
            raise ValueError(f"Unknown scale: {self.scale}")

        label = int(self.labels[idx])

        return torch.from_numpy(frames), label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["file"] = None
        return state

    def __del__(self):
        f = getattr(self, "file", None)
        if f is not None:
            f.close()


def make_loaders(
    root="./data/shd",
    T=16,
    batch_size=64,
    num_workers=0,
    scale="max",
    seed=42
):
    root = Path(root)

    train_all = SHDDataset(
        root / "shd_train.h5", T=T, scale=scale
    )
    test_set = SHDDataset(
        root / "shd_test.h5", T=T, scale=scale
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(train_all), generator=generator
    ).tolist()

    n_val = int(0.1 * len(indices))
    val_set = Subset(train_all, indices[:n_val])
    train_set = Subset(train_all, indices[n_val:])

    def loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    return (
        loader(train_set, True),
        loader(val_set, False),
        loader(test_set, False)
    )


if __name__ == "__main__":
    train, val, test = make_loaders(batch_size=4)

    for name, loader in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:
        x, y = next(iter(loader))
        print(name, x.shape, y.shape)
        print("min/max:", x.min().item(), x.max().item())
        print("labels:", y.tolist())
        assert x.shape == (4, 16, 700)
        assert y.min() >= 0 and y.max() < 20

    print("SHD DATA TEST PASSED")