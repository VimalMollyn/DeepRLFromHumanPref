import numpy as np
import torch
import torch.nn as nn


class FullyConnectedMLP(nn.Module):
    """Vanilla two hidden layer multi-layer perceptron for reward prediction."""

    def __init__(self, obs_shape, act_shape, h_size=64, dropout=0.5):
        super().__init__()
        input_dim = int(np.prod(obs_shape) + np.prod(act_shape))

        self.model = nn.Sequential(
            nn.Linear(input_dim, h_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(h_size, 1),
        )

    def forward(self, obs, act):
        """
        Forward pass through the reward predictor.

        Args:
            obs: Observation tensor of shape (batch, *obs_shape) or (batch, seq, *obs_shape)
            act: Action tensor of shape (batch, *act_shape) or (batch, seq, *act_shape)

        Returns:
            Predicted reward tensor
        """
        # Flatten observations and actions if needed
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=-len(obs.shape) + 2)
        if act.dim() > 2:
            act = act.flatten(start_dim=-len(act.shape) + 2)

        x = torch.cat([obs, act], dim=-1)
        return self.model(x)
