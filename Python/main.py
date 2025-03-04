import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from env import exePath
from Actor import Actor
from Critic import Critic
from torch.optim.lr_scheduler import StepLR

# 📌 Définition du device (GPU si dispo)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
torch.backends.cudnn.benchmark = True

# ✅ Définition des observations
OBSERVATION_KEYS = {
    "head_position": slice(0, 3),
    "head_rotation": slice(3, 6),
    "left_hip_joint_angle": {"x": 6, "z": 7},
    "right_hip_joint_angle": {"x": 8, "z": 9},
    "joint_angles": {"left_tibia": 10, "left_foot": 11, "right_tibia": 12, "right_foot": 13},
    "feet_contact": {"left_foot_grounded": 14, "right_foot_grounded": 15},
    "objectiv_direction": 16,
    "angle_observed": 17
}

input_dim = 18  # Taille des observations
action_dim = 8   # Nombre d'actions

# ✅ Fonction pour charger les modèles (évite l'erreur si absent)
def load_model(model, filename):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename, map_location=device))
        print(f"✅ Modèle {filename} chargé !")
    else:
        print(f"⚠️ Pas de modèle trouvé pour {filename}, démarrage vierge.")

# ✅ Initialisation des modèles
actor = Actor(input_dim, action_dim).to(device)
critic = Critic(input_dim).to(device)

# Chargement si dispo
load_model(actor, "saveActor.pth")
load_model(critic, "saveCritic.pth")

# ✅ Optimizers + Scheduler
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
actor_scheduler = StepLR(actor_optimizer, step_size=500, gamma=0.9)
critic_scheduler = StepLR(critic_optimizer, step_size=500, gamma=0.9)

# ✅ Chargement de l'environnement Unity
channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=50.0)
env = UnityEnvironment(file_name=exePath, side_channels=[channel], worker_id=3, no_graphics=True)

# ✅ Initialisation
env.reset()
behavior_name = list(env.behavior_specs)[0]
action_spec = env.behavior_specs[behavior_name].action_spec
gamma = 0.99  # Facteur de discount

num_episodes = 999999999999  # Nombre total d'épisodes
reward_history = []  # Stockage des scores

plt.ion()  # Mode interactif pour le graphe

for episode in range(num_episodes):
    print(f"Démarrage de l'épisode {episode}")
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    num_actions = action_spec.continuous_size

    observations, actions_list, rewards, values = [], [], [], []

    while len(terminal_steps) == 0:
        if len(decision_steps) > 0:
            for agent_id in decision_steps.agent_id:
                obs = np.array(decision_steps[agent_id].obs, dtype=np.float32).flatten()
                if len(obs) != input_dim:
                    print(f"⚠️ Alerte : Taille incorrecte ({len(obs)} vs {input_dim})")
                    continue

                ob_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_dist = actor(ob_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum()

                actions = action.detach().cpu().numpy()
                noise = np.random.normal(0, 0.1, actions.shape)
                actions = np.clip(actions + noise, -1, 1)

                value = critic(ob_tensor).detach().cpu().numpy()

                observations.append(ob_tensor)
                actions_list.append(action.clone().detach())  # ✅ Correction ici
                values.append(value)

            if num_actions > 0:
                try:
                    actions = np.array(actions).reshape((len(decision_steps), num_actions))
                    actions_tuple = ActionTuple(continuous=actions)
                except ValueError as e:
                    print(f"Erreur reshape actions: {e}")
                    continue
            else:
                actions_tuple = ActionTuple(discrete=np.zeros((len(decision_steps), action_spec.discrete_size), dtype=np.int32))

            if len(decision_steps) > 0:
                env.set_actions(behavior_name, actions_tuple)

        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(terminal_steps) > 0:
            for agent_id in terminal_steps.agent_id:
                rewards.append(terminal_steps[agent_id].reward)

    # ✅ Calcul des avantages
    returns = np.zeros_like(rewards, dtype=np.float32)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)  # ✅ Fix taille
    values = torch.tensor(np.array(values), dtype=torch.float32, device=device).unsqueeze(1)  # ✅ Fix taille
    values.requires_grad_()
    advantages = returns - values

    # ✅ Mise à jour de l'Actor
    actor_loss = []
    for ob, action, advantage in zip(observations, actions_list, advantages):
        action_tensor = torch.tensor(action, dtype=torch.float32, device=device).clone().detach()  # ✅ Fix ici
        log_prob = actor(ob).log_prob(action_tensor).sum()
        actor_loss.append(-log_prob * advantage)

    actor_loss = torch.stack(actor_loss).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # ✅ Mise à jour du Critic avec plusieurs itérations
    critic_epochs = 5  # Mise à jour plus fréquente du Critic
    for _ in range(critic_epochs):
        critic_loss = F.mse_loss(values, returns)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    actor_scheduler.step()
    critic_scheduler.step()

    # ✅ Stockage du score et mise à jour du graphe
    total_reward = sum(rewards)
    reward_history.append(total_reward)

    if episode % 10 == 0:  # Mise à jour du graphique tous les 10 épisodes
        plt.clf()
        plt.plot(reward_history)
        plt.xlabel("Épisode")
        plt.ylabel("Score")
        plt.title("Progression de l'apprentissage")
        plt.pause(0.01)

    print(f"Episode {episode}, Actor Loss: {actor_loss.item():.3f}, Critic Loss: {critic_loss.item():.3f}")
    print(f"Score de l'épisode : {float(terminal_steps.reward[0])}")

    # ✅ Sauvegarde tous les 100 épisodes
    if episode % 100 == 0:
        print("💾 Sauvegarde...")
        torch.save(actor.state_dict(), "saveActor.pth")
        torch.save(critic.state_dict(), "saveCritic.pth")

env.close()
print("🎉 Terminé !")
