from torch import nn


class LinearModel(nn.Module):
    def __init__(self, input_size, bias):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1, bias=bias),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
