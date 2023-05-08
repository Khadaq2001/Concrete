import torch
from torch import nn

class ConcreteAutoencoder(nn.Module):
    def __init__(self, input_dim, k, hidden_dim,device):
        super().__init__()
        self.device = device
        self.alpha = nn.Parameter(torch.rand(input_dim, k))
        self.decoder = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax(self, alpha, temperature):
        y = alpha + self.sample_gumbel(alpha.size()).to(self.device)
        return nn.functional.softmax(y / temperature, dim = 0)
    
    def forward(self, x, temperature):
        m = self.gumbel_softmax(self.alpha, temperature)
        z = torch.mm(x,m)
        reconstruction = self.decoder(z)
        return reconstruction, z, m


class FullConnectedLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, device):
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z
    
