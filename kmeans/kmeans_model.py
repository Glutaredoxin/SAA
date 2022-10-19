import torch.nn as nn

class kmeans_model(nn.Module):
    def __init__(self):
        super(kmeans_model, self).__init__()
        self.linear1 = nn.Linear(640, 100, bias=True)
        self.relu = nn.LeakyReLU(0.1)
        self.linear2 = nn.Linear(100, 100, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x