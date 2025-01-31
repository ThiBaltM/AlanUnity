
#Le rôle de l’acteur est de transformer les obsevrations en actions continues (dans l’intervalle [-1, 1])
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # Couche d'entrée
            nn.ReLU(),
            nn.Linear(64, 64),        # Couche cachée
            nn.ReLU(),
            nn.Linear(64, action_dim), # Couche de sortie
            nn.Tanh()                 # Sortie dans [-1, 1]
        )

    def forward(self, x):
        return self.fc(x)