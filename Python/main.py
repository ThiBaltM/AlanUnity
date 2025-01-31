import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from env import exePath
from Actor import Actor
from Critic import Critic
import torch
import torch.nn.functional as F
def extract_extra_features(structured_ob):
    """
    Convertit les objectifs et pénalités en features normalisées pour le Critique.

    :param structured_ob: Dictionnaire contenant les observations structurées.
    :return: Tensor PyTorch des objectifs et pénalités.
    """
    extra_features = [
        float(structured_ob["objectives"]["is_standing"]),  # 1 si debout, 0 sinon
        float(structured_ob["objectives"]["moving_forward"]),  # 1 si avance, 0 sinon
        structured_ob["objectives"]["distance_traveled"],  # Distance parcourue
        float(structured_ob["penalties"]["has_fallen"]),  # 1 si tombé, 0 sinon
        float(structured_ob["penalties"]["excessive_oscillations"])  # 1 si oscille trop, 0 sinon
    ]
    return torch.FloatTensor(extra_features).unsqueeze(0)  # Convertir en tensor


    #create dictionnary : 
OBSERVATION_KEYS = {
    "head_position": slice(0, 3),      # 3 valeurs : (x, y, z)
    "head_rotation": slice(3, 6),      # 3 valeurs : (yaw, pitch, roll)
    "joint_angles": {
        "left_hip": 6, "left_tibia": 7, "left_foot": 8,
        "right_hip": 9, "right_tibia": 10, "right_foot": 11
    },
    "feet_contact": {
        "left_foot_grounded": 12, "right_foot_grounded": 13
    }
}

#init input : 

input_dim = 14  # Nombre d'observations
action_dim = 8  # Nombre d'actions (contrôle des articulations)

# Création des modèles
actor = Actor(input_dim, action_dim)
critic = Critic(input_dim)

# Configurer le canal de configuration du moteur
channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=1.0)

# Charger l'environnement Unity
env = UnityEnvironment(file_name=exePath,side_channels=[channel], worker_id=3, no_graphics=True)

# Définir le nombre d'épisodes
num_episodes = 1000
print("test1")
# Obtenir le comportement par défaut
env.reset()
print("test2")

behavior_name = list(env.behavior_specs)[0]
# Obtenir l'ActionSpec du comportement
action_spec = env.behavior_specs[behavior_name].action_spec

print(f"Started script")
# Entrainer l'agent
for episode in range(num_episodes):
    print(f"Démarrage de l'épisode {episode}")
    env.reset()
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    while len(terminal_steps) == 0:
        if len(decision_steps) > 0:
            # Collecter des observations
            for agent_id in decision_steps.agent_id:
                observations = np.array(decision_steps[agent_id].obs, dtype=np.float32)

                ob = torch.tensor(observations).flatten()
                prev_state = None

                structured_ob = {
                                "head_position": ob[OBSERVATION_KEYS["head_position"]].tolist(),
                                "head_rotation": ob[OBSERVATION_KEYS["head_rotation"]].tolist(),
                                "joint_angles": {key: ob[idx].item() for key, idx in OBSERVATION_KEYS["joint_angles"].items()},
                                "feet_contact": {key: bool(int(ob[idx].item())) for key, idx in OBSERVATION_KEYS["feet_contact"].items()}
                            }
                if prev_state:
                    prev_head_position = prev_state["head_position"]
                    prev_joint_angles = prev_state["joint_angles"]
                else:
                    prev_head_position = structured_ob["head_position"]
                    prev_joint_angles = structured_ob["joint_angles"]

                # Calcul des objectifs positifs
                structured_ob["objectives"] = {
                                                "is_standing": structured_ob["head_position"][1] > 1.0,  # Rester debout (y > 1.0)
                                                "moving_forward": structured_ob["head_position"][2] > prev_head_position[2],  # Avancer (Z augmente)
                                                "distance_traveled": abs(structured_ob["head_position"][2] - prev_head_position[2])  # Distance parcourue
                                            }
                # Calcul des pénalités
                structured_ob["penalties"] = {
                    "has_fallen": structured_ob["head_position"][1] < 0.5,  # Tomber (y < 0.5)
                    "excessive_oscillations": any(
                        abs(structured_ob["joint_angles"][key] - prev_joint_angles[key]) > 5.0
                        for key in structured_ob["joint_angles"]
                    )
                    }

                    # Calculer une récompense totale
                reward = 0

                # Récompenses positives
                if structured_ob["objectives"]["is_standing"]:
                    reward += 1
                if structured_ob["objectives"]["moving_forward"]:
                    reward += 10
                reward += structured_ob["objectives"]["distance_traveled"]
                # Pénalités
                if structured_ob["penalties"]["has_fallen"]:
                    reward -= 10
                if structured_ob["penalties"]["excessive_oscillations"]:
                    reward -= 1

                # Ajouter la récompense totale au dictionnaire
                structured_ob["reward"] = reward

                print(f"Observations de l'agent {agent_id} :", structured_ob)
                # Extraire toutes les valeurs numériques et les convertir en Tensor
                ob_tensor = torch.FloatTensor(
                    structured_ob["head_position"] +
                    structured_ob["head_rotation"] +
                    list(structured_ob["joint_angles"].values()) +
                    list(structured_ob["feet_contact"].values())
                ).unsqueeze(0)  # Ajoute une dimension batch

                # Vérifier si l'entrée a bien la bonne taille avant de passer à l'Acteur
                assert ob_tensor.shape[1] == input_dim, f"Erreur: ob_tensor a {ob_tensor.shape[1]} dimensions, attendu {input_dim}"

                # Passer l'observation corrigée à l'Acteur
                action = actor(ob_tensor).detach().numpy()
                if action is not None:
                    noise = np.random.normal(0, 0.1, action.shape)  # Ajouter un peu de bruit aléatoire
                    action = np.clip(action + noise, -1, 1)  # Assurer que l'action reste entre [-1,1]

                print("Action générée :", action)

                extra_features = extract_extra_features(structured_ob)
                # Vérifier la taille du tensor avant de passer au Critic
                print(f"ob_tensor shape: {ob_tensor.shape}, extra_features shape: {extra_features.shape}")

                # Passer l'état et les features supplémentaires au Critic
                value = critic(ob_tensor, extra_features).detach().numpy()
                print("Valeur de l'état :", value.item())

                """action = actor(ob)
                value = critic(ob)

                print("Action générée :", action.detach().numpy())
                print("Valeur de l'état :", value.item())"""

            """ print(f"Observations de l'épisode {episode} pour l'agent {agent_id}:")
                for i, obs in enumerate(observations):
                    for o in obs:
                        print(round(o,2), " | ")
                    print(f"Observation {i}: {obs}")"""
        # Initialiser des actions par défaut (zéro) avec les bonnes dimensions
        actions = [[0] * action_spec.continuous_size] * len(decision_steps)

        # Envoyer les actions à Unity
        #env.set_actions(behavior_name, actions)

        env.step()

        # Mettre à jour les steps
        decision_steps, terminal_steps = env.get_steps(behavior_name)

env.close()
