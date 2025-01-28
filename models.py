from torch import nn


class LinearModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        bias: bool = True
    ) -> None:

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, output_size, bias=bias),
        )
    
    def forward(self, x):
        return self.layers(x)


class SimpleFFN(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int
    ) -> None:

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.layers(x)
