import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from env import exePath
from Actor import Actor
from Critic import Critic
import torch
import torch.nn.functional as F
from visualizer import visualize_network_dynamic

def extract_extra_features(structured_ob):
    """
    Convertit les objectifs et pénalités en features normalisées pour le Critique.
    """
    extra_features = [
        float(structured_ob["objectives"]["is_standing"]),
        float(structured_ob["objectives"]["moving_forward"]),
        structured_ob["objectives"]["distance_traveled"],
        float(structured_ob["penalties"]["has_fallen"]),
        float(structured_ob["penalties"]["excessive_oscillations"])
    ]
    return torch.FloatTensor(extra_features).unsqueeze(0)

OBSERVATION_KEYS = {
    "head_position": slice(0, 3),
    "head_rotation": slice(3, 6),
    "joint_angles": {
        "left_hip": 6, "left_tibia": 7, "left_foot": 8,
        "right_hip": 9, "right_tibia": 10, "right_foot": 11
    },
    "feet_contact": {
        "left_foot_grounded": 12, "right_foot_grounded": 13
    }
}

input_dim = 14  # Nombre d'observations
action_dim = 8  # Nombre d'actions (contrôle des articulations)

actor = Actor(input_dim, action_dim)
critic = Critic(input_dim)

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1.0)

env = UnityEnvironment(file_name=exePath, side_channels=[channel], worker_id=3, no_graphics=False)

num_episodes = 1000
env.reset()
behavior_name = list(env.behavior_specs)[0]
action_spec = env.behavior_specs[behavior_name].action_spec

print(f"Started script - ActionSpec: {action_spec}")

screen, clock, layers = None, None, []

for episode in range(num_episodes):
    print(f"Démarrage de l'épisode {episode}")
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    
    num_actions = action_spec.continuous_size
    print(f"Num actions (continuous): {num_actions}, Discrete branches: {action_spec.discrete_size}")

    if num_actions == 0 and action_spec.discrete_size > 0:
        print("⚠️ Cet agent utilise des actions discrètes et non continues !")

    while len(terminal_steps) == 0:
        if len(decision_steps) > 0:
            for agent_id in decision_steps.agent_id:
                observations = np.array(decision_steps[agent_id].obs, dtype=np.float32).flatten()
                ob = torch.tensor(observations)

                structured_ob = {
                    "head_position": ob[OBSERVATION_KEYS["head_position"]].tolist(),
                    "head_rotation": ob[OBSERVATION_KEYS["head_rotation"]].tolist(),
                    "joint_angles": {key: ob[idx].item() for key, idx in OBSERVATION_KEYS["joint_angles"].items()},
                    "feet_contact": {key: bool(int(ob[idx].item())) for key, idx in OBSERVATION_KEYS["feet_contact"].items()}
                }

                structured_ob["objectives"] = {
                    "is_standing": structured_ob["head_position"][1] > 1.0,
                    "moving_forward": structured_ob["head_position"][2] > 0,
                    "distance_traveled": structured_ob["head_position"][2]
                }

                structured_ob["penalties"] = {
                    "has_fallen": structured_ob["head_position"][1] < 0.5,
                    "excessive_oscillations": any(
                        abs(structured_ob["joint_angles"][key]) > 5.0
                        for key in structured_ob["joint_angles"]
                    )
                }

                reward = 0
                if structured_ob["objectives"]["is_standing"]:
                    reward += 1
                if structured_ob["objectives"]["moving_forward"]:
                    reward += 10
                reward += structured_ob["objectives"]["distance_traveled"]
                if structured_ob["penalties"]["has_fallen"]:
                    reward -= 10
                if structured_ob["penalties"]["excessive_oscillations"]:
                    reward -= 1

                structured_ob["reward"] = reward
                print(f"Observations de l'agent {agent_id} :", structured_ob)

                ob_tensor = torch.FloatTensor(
                    structured_ob["head_position"] +
                    structured_ob["head_rotation"] +
                    list(structured_ob["joint_angles"].values()) +
                    list(structured_ob["feet_contact"].values())
                ).unsqueeze(0)

                assert ob_tensor.shape[1] == input_dim, f"Erreur: ob_tensor a {ob_tensor.shape[1]} dimensions, attendu {input_dim}"

                actions = actor(ob_tensor).detach().numpy()
                if actions is not None:
                    noise = np.random.normal(0, 0.1, actions.shape)
                    actions = np.clip(actions + noise, -1, 1)

                activations = []
                x = ob_tensor
                for layer in actor.fc:
                    if isinstance(layer, torch.nn.Linear):
                        x = F.relu(layer(x))
                        activations.append(x.detach().numpy()[0].tolist())

                if not layers:
                    layers = [[] for _ in range(len(actor.fc) if hasattr(actor, 'fc') else 0)]

                screen, clock, layers = visualize_network_dynamic(actor, activations, screen, clock)

                print("Action générée :", actions)

                extra_features = extract_extra_features(structured_ob)
                print(f"ob_tensor shape: {ob_tensor.shape}, extra_features shape: {extra_features.shape}")

                value = critic(ob_tensor, extra_features).detach().numpy()
                print("Valeur de l'état :", value.item())
            print(num_actions,"ahhh\n\n")
            if num_actions > 0:
                try:
                    actions = np.array(actions).reshape((len(decision_steps), num_actions))
                    actions_tuple = ActionTuple(continuous=actions)
                except ValueError as e:
                    print(f"Erreur lors du reshape des actions: {e}")
                    continue
            else:
                actions_tuple = ActionTuple(discrete=np.zeros((len(decision_steps), action_spec.discrete_size), dtype=np.int32))

            print(f"Nombre d'agents : {len(decision_steps)}, Nombre d'actions continues attendues : {num_actions}")

            if len(decision_steps) > 0:
                env.set_actions(behavior_name, actions_tuple)
            else:
                print("⚠️ Aucun agent actif, pas d'actions à envoyer.")

        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

env.close()
