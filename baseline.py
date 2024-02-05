import torch

class baseline(torch.nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y = self.linear(x)
        return y