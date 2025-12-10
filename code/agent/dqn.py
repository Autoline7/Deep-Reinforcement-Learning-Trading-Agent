import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, lr=1e-4):
        """
        Deep Q-Network:
        input_dim: dimension of the state vector
        output_dim: number of actions (3: Buy, Sell, Hold)
        """
        super(DQN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, output_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, state):
        """
        Forward pass through the network.
        state: tensor shape [batch_size, input_dim]
        """
        return self.model(state)
