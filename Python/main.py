import numpy as np
import torch
import torch.nn.functional as F
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from env import exePath
from Actor import Actor
from Critic import Critic
from visualizer import visualize_network_dynamic

# Définir l'appareil (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark = True



OBSERVATION_KEYS = {
    "head_position": slice(0, 3),
    "head_rotation": slice(3, 6),
    "left_hip_joint_angle": {"x": 6, "z": 7},
    "right_hip_joint_angle": {"x": 8, "z": 9},
    "joint_angles": {
        "left_tibia": 10, "left_foot": 11,
        "right_tibia": 12, "right_foot": 13
    },
    "feet_contact": {"left_foot_grounded": 14, "right_foot_grounded": 15},
    "objectiv_direction": 16,
    "angle_observed": 17
}

input_dim = 18
action_dim = 8

# Charger les modèles sur GPU
actor = Actor(input_dim, action_dim).to(device)
critic = Critic(input_dim).to(device)
actor.load("saveActor")
critic.load("saveCritic")
actor.train()
critic.train()

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4,fused=True)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3,fused=True)

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1.0)

env = UnityEnvironment(file_name=exePath, side_channels=[channel], worker_id=3, no_graphics=False)

env.reset()
behavior_name = list(env.behavior_specs)[0]
action_spec = env.behavior_specs[behavior_name].action_spec

print(f"Started script - ActionSpec: {action_spec}")

gamma = 0.99
num_episodes = 99999999910000

for episode in range(num_episodes):
    print(f"Démarrage de l'épisode {episode}")
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    num_actions = action_spec.continuous_size

    observations = []
    actions_list = []
    rewards = []
    values = []

    while len(terminal_steps) == 0:
        if len(decision_steps) > 0:
            for agent_id in decision_steps.agent_id:
                obs = np.array(decision_steps[agent_id].obs, dtype=np.float32).flatten()
                if len(obs) != input_dim:
                    print(f"⚠️ Alerte : Nombre d'observations incorrect ({len(obs)} vs attendu {input_dim})")
                    continue

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ob = torch.tensor(obs, dtype=torch.float32, device=device)
                ob_tensor = ob.unsqueeze(0)

                assert ob_tensor.shape[1] == input_dim, f"Erreur: ob_tensor a {ob_tensor.shape[1]} dimensions, attendu {input_dim}"

                action_dist = actor(ob_tensor)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum()
                
                actions = action.detach().cpu().numpy()
                noise = np.random.normal(0, 0.1, actions.shape)
                actions = np.clip(actions + noise, -1, 1)

                value = critic(ob_tensor).detach().cpu().numpy()

                observations.append(ob_tensor)
                actions_list.append(action)
                values.append(value)

            if num_actions > 0:
                try:
                    actions = np.array(actions).reshape((len(decision_steps), num_actions))
                    actions_tuple = ActionTuple(continuous=actions)
                except ValueError as e:
                    print(f"Erreur lors du reshape des actions: {e}")
                    continue
            else:
                actions_tuple = ActionTuple(discrete=np.zeros((len(decision_steps), action_spec.discrete_size), dtype=np.int32))

            if len(decision_steps) > 0:
                env.set_actions(behavior_name, actions_tuple)
            else:
                print("⚠️ Aucun agent actif, pas d'actions à envoyer.")

        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(terminal_steps) > 0:
            for agent_id in terminal_steps.agent_id:
                rewards.append(terminal_steps[agent_id].reward)

    # Calcul des avantages
    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device, requires_grad=True)
    values = torch.tensor(values, dtype=torch.float32, device=device, requires_grad=True).squeeze()
    advantages = returns - values

    # Mise à jour de l'acteur
    actor_loss = []
    for ob, action, advantage in zip(observations, actions_list, advantages):
        action_tensor = torch.tensor(action, dtype=torch.float32, device=device, requires_grad=True)
        log_prob = actor(ob).log_prob(action_tensor).sum()
        actor_loss.append(-log_prob * advantage)

    actor_loss = torch.stack(actor_loss).sum()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Mise à jour du critique
    critic_loss = F.mse_loss(values, returns)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    print(f"Episode {episode}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")
    print(f"Score de l'épisode : {float(terminal_steps.reward[0])}")

    if episode % 100 == 0:
        print("Sauvegarde...")
        actor.save("saveActor")
        critic.save("saveCritic")
        print("Sauvegarde terminée !")

env.close()
print("Terminé !")