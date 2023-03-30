import torch 
from torch import nn

class ConcreteAutoencode(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,
                 num_sample, temperature):
        super().__init__()
        self.num_sample = num_sample
        self.temperature = temperature

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.logits = nn.Parameter(torch.randn(latent_dim, num_sample))

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U+eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return nn.functional.softmax(y/temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature):
        y = self.gumbel_softmax_sample (logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard = y_hard.view(*shape)
        return(y_hard -y).detach() + y
    
    def forward(self,x):
        latent = self.encoder(x)
        logits = torch.mm(latent, self.logits)
        z = self.gumbel_softmax(logits, self.temperature)
        reconstruction = self.decoder(z)
        return reconstruction, z