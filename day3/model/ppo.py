import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, action_dim=7, state_dependent_sigma=True):
        super().__init__()
        # actor trunk
        self.actor_mlp = nn.Sequential(
            nn.Linear(19, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64),  nn.ELU(),
        )
        # heads
        self.mu = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

        # sigma learning
        self.state_dependent_sigma = state_dependent_sigma
        if state_dependent_sigma:
            self.sigma_head = nn.Linear(64, action_dim)   # outputs log_std
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))  # global log_std

        # activations (이미지 구조와 동일)
        self.mu_act = nn.Identity()
        self.value_act = nn.Identity()
        self.sigma_act = nn.Identity()

    def forward(self, x):
        h = self.actor_mlp(x)
        mu = self.mu_act(self.mu(h))
        v  = self.value_act(self.value(h))

        if self.state_dependent_sigma:
            log_std = self.sigma_head(h)
            log_std = torch.clamp(log_std, -5.0, 2.0)  # 안정화
            sigma = self.sigma_act(torch.exp(log_std))
        else:
            sigma = self.sigma_act(self.log_std.exp()).expand_as(mu)

        return mu, sigma, v