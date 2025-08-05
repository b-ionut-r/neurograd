from neurograd import Tensor, float32
import math
import random

class Dataset:
    def __init__(self, X, y, dtype = float32):
        assert len(X) == len(y), "Mismatched input and label lengths"
        self.X = Tensor(X, dtype=dtype)
        self.y = Tensor(y, dtype=dtype)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    def shuffle(self, seed: int = 42):
        indices = list(range(len(self)))
        random.seed(seed)
        random.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
    def __repr__(self):
        return f"<Dataset: {len(self)} samples, dtype={self.X.data.dtype}>"
    def __str__(self):
        preview_x = self.X[:1]
        preview_y = self.y[:1]
        return (f"Dataset:\n"
                f"  Total samples: {len(self)}\n"
                f"  Input preview: {preview_x}\n"
                f"  Target preview: {preview_y}")
        

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, 
                 shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.dataset))
        return self.dataset[start:end]
    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle(self.seed)
        for idx in range(len(self)):
            yield self[idx]
    def __repr__(self):
        return (f"<DataLoader: {len(self)} batches, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, seed={self.seed}>")
    def __str__(self):
        return (f"DataLoader:\n"
                f"  Batches: {len(self)}\n"
                f"  Batch size: {self.batch_size}\n"
                f"  Shuffle: {self.shuffle}\n"
                f"  Seed: {self.seed}")

                     


