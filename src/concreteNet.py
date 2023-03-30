import torch
from torch import nn


class ConcreteAutoencoder(nn.Module):
    def __init__(self, input_dim, k, hidden_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.decoder = nn.Sequential(
            nn.Linear(k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def sample_gumbel(self, shape, eps=1e-5):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return nn.functional.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward(self, x):
        sample_size = x.shape[1]
        alpha = nn.Parameter(torch.randn(self.k, sample_size))
        logits = torch.mm(x, alpha)
        z = self.gumbel_softmax_sample(logits, self.temperature)
        reconstruction = self.decoder(z)
        return reconstruction, z
