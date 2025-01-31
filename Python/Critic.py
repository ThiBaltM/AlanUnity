import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, input_dim, extra_dim=5):
        super(Critic, self).__init__()
        total_input_dim = input_dim + extra_dim  # Assurez-vous que cette ligne existe !
        
        self.fc = nn.Sequential(
            nn.Linear(total_input_dim, 128),  # Ici, total_input_dim doit être correct
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Sortie scalaire
        )

    def forward(self, state, extra_features):
        x = torch.cat([state, extra_features], dim=-1)  # Concaténer l'entrée
        return self.fc(x)

