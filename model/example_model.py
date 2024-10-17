import torch as T

class ExampleModel(T.nn.Module):
    def __init__(self, n_neurons: int):
        super().__init__()
        self.net = T.nn.Sequential(
            T.nn.Linear(1, n_neurons),
            T.nn.ReLU(),
            T.nn.Linear(n_neurons, n_neurons),
            T.nn.ReLU(),
            T.nn.Linear(n_neurons, 1)
        )

    def forward(self, x):
        return self.net(x)