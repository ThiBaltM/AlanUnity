import torch

class PPO:
    def __init__(self, actor, critic, gamma=0.99, epsilon=0.2, learning_rate=0.001):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def accumulate_trajectory(self, env, behavior_name):
        """
        Accumule les trajectoires pendant un épisode
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        done = False
        state = env.reset()  # Réinitialisation de l'environnement
        while not done:
            action = self.actor(state)  # L'acteur choisit une action
            next_state, reward, done, _ = env.step(action)  # L'environnement renvoie la nouvelle observation, récompense, etc.
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
        
        return states, actions, rewards, next_states, dones

    def update(self, states, actions, rewards, next_states, dones):
        """
        Met à jour l'acteur et le critique après l'épisode
        """
        # Calcul des valeurs d'état pour chaque état
        state_values = self.critic(states)
        next_state_values = self.critic(next_states)
        
        # Calcul de la cible (retour) pour chaque état
        targets = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Calcul de l'avantage
        advantages = targets - state_values
        
        # Mise à jour du critique
        critic_loss = torch.mean((targets - state_values) ** 2)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # Calcul de la perte de l'acteur (perte de l'avantage)
        action_probs = self.actor(states)
        old_action_probs = action_probs.detach()
        
        ratio = action_probs / old_action_probs
        actor_loss = -torch.mean(ratio * advantages)  # Maximiser l'avantage
        clipped_actor_loss = torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages))
        
        self.optimizer_actor.zero_grad()
        clipped_actor_loss.backward()
        self.optimizer_actor.step()
