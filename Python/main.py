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

OBSERVATION_KEYS = {
    "head_position": slice(0, 3),
    "head_rotation": slice(3, 6),
    "left_hip_joint_angle": {
        "x": 6, "z": 7
    },
    "right_hip_joint_angle": {
        "x": 8, "z": 9
    },
    "joint_angles": {
        "left_tibia": 10, "left_foot": 11,
        "right_tibia": 12, "right_foot": 13
    },
    "feet_contact": {
        "left_foot_grounded": 14, "right_foot_grounded": 15
    },
    "objectiv_direction": 16  # Correction finale de l'index
}

input_dim = 17  # Assuré que cela correspond au nombre d'observations attendu
action_dim = 8

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

    while len(terminal_steps) == 0:
        if len(decision_steps) > 0:
            for agent_id in decision_steps.agent_id:
                observations = np.array(decision_steps[agent_id].obs, dtype=np.float32).flatten()                
                if len(observations) != input_dim:
                    print(f"⚠️ Alerte : Nombre d'observations incorrect ({len(observations)} vs attendu {input_dim})")
                    continue  # On ignore cette itération pour éviter l'erreur
                
                ob = torch.tensor(observations)

                structured_ob = {
                    "head_position": ob[OBSERVATION_KEYS["head_position"]].tolist(),
                    "head_rotation": ob[OBSERVATION_KEYS["head_rotation"]].tolist(),
                    "left_hip_joint_angle": {key: ob[idx].item() for key, idx in OBSERVATION_KEYS["left_hip_joint_angle"].items()},
                    "right_hip_joint_angle": {key: ob[idx].item() for key, idx in OBSERVATION_KEYS["right_hip_joint_angle"].items()},
                    "joint_angles": {key: ob[idx].item() for key, idx in OBSERVATION_KEYS["joint_angles"].items()},
                    "feet_contact": {key: bool(int(ob[idx].item())) for key, idx in OBSERVATION_KEYS["feet_contact"].items()},
                    "objectiv_direction": ob[OBSERVATION_KEYS["objectiv_direction"]].item()
                }

                ob_tensor = torch.FloatTensor(
                    structured_ob["head_position"] +
                    structured_ob["head_rotation"] +
                    list(structured_ob["left_hip_joint_angle"].values()) +
                    list(structured_ob["right_hip_joint_angle"].values()) +
                    list(structured_ob["joint_angles"].values()) +
                    list(structured_ob["feet_contact"].values()) +
                    [structured_ob["objectiv_direction"]]
                ).unsqueeze(0)

                assert ob_tensor.shape[1] == input_dim, f"Erreur: ob_tensor a {ob_tensor.shape[1]} dimensions, attendu {input_dim}"

                actions = actor.fc(ob_tensor).detach().numpy()
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

                value = critic(ob_tensor).detach().numpy()

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
    print('score de lépisode : '+str(float(terminal_steps.reward[0])))
env.close()
