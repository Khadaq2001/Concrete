import torch
from torch import nn

class ConcreteAutoencoder(nn.Module):
    def __init__(self, input_dim, k, hidden_dim,device, temperature=0.1):
        super().__init__()
        self.temperature = temperature
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

    def gumbel_softmax_sample(self, alpha, temperature):
        y = alpha + self.sample_gumbel(alpha.size()).to(self.device)
        return nn.functional.softmax(y / temperature, dim = 0)

    def gumbel_softmax(self, alpha, temperature):
        y = self.gumbel_softmax_sample(alpha, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward(self, x):
        m = self.gumbel_softmax_sample(self.alpha, self.temperature)
        z = torch.mm(x,m)
        reconstruction = self.decoder(z)
        return reconstruction, z, m
