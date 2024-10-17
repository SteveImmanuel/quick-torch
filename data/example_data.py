import torch as T

class ExampleData(T.utils.data.Dataset):
    def __init__(self, n_samples: int, val_ratio: float, validation: bool = False):
        super().__init__()

        x  = T.randn(n_samples).unsqueeze(1)
        y = x ** 2
        split_idx = int(n_samples * val_ratio)
        if validation:
            self.x = x[:split_idx]
            self.y = y[:split_idx]
        else:
            self.x = x[split_idx:]
            self.y = y[split_idx:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
