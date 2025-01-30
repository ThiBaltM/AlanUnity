import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from env import exePath


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
                observations = decision_steps[agent_id].obs
                print(f"Observations de l'épisode {episode} pour l'agent {agent_id}:")
                for i, obs in enumerate(observations):
                    for o in obs:
                        print(round(o,2), " | ")
                    print(f"Observation {i}: {obs}")
        # Initialiser des actions par défaut (zéro) avec les bonnes dimensions
        actions = [[0] * action_spec.continuous_size] * len(decision_steps)

        # Envoyer les actions à Unity
        #env.set_actions(behavior_name, actions)

        env.step()

        # Mettre à jour les steps
        decision_steps, terminal_steps = env.get_steps(behavior_name)

env.close()
