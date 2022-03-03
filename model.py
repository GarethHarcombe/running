from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(            
            nn.Linear(60 * 60 * 6 * 4, 60 * 60),
            nn.ReLU(),
            nn.Linear(60 * 60, 60),
            nn.ReLU(),
            nn.Linear(60, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits