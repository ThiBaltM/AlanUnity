def visualize_network_dynamic(actor, activations=None, screen=None, clock=None):
    import pygame
    import networkx as nx
    import numpy as np
    import torch

    # 📌 Paramètres d'affichage
    WIDTH, HEIGHT = 1200, 800
    NODE_RADIUS = 15
    LAYER_SPACING = WIDTH // 6  # Ajusté dynamiquement plus tard
    NEURON_SPACING = HEIGHT // 10  # Ajusté dynamiquement plus tard

    # 🎨 Couleurs
    TRANSPARENT = (255, 255, 255)
    BLUE = (30, 144, 255)
    GREEN = (0, 255, 0)
    RED = (255, 69, 0)
    GRAY = (180, 180, 180)
    DARK_GRAY = (100, 100, 100)

    # 📌 Initialiser Pygame
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Visualisation Dynamique du Réseau PPO")

    if clock is None:
        clock = pygame.time.Clock()

    # 📌 Créer le graphe du réseau
    def create_graph(model):
        graph = nx.DiGraph()
        layers = []
        prev_nodes = []

        for layer_num, layer in enumerate(model.fc):
            if isinstance(layer, torch.nn.Linear):
                in_features, out_features = layer.weight.shape
                current_nodes = []

                # Ajouter les neurones de la couche actuelle
                for i in range(out_features):
                    node_name = f"{layer_num}_{i}"
                    graph.add_node(node_name, layer=layer_num)
                    current_nodes.append(node_name)

                # Ajouter les connexions avec la couche précédente
                if prev_nodes:
                    for prev_node in prev_nodes:
                        for curr_node in current_nodes:
                            weight = np.random.uniform(-1, 1)  # Poids simulés
                            graph.add_edge(prev_node, curr_node, weight=weight)

                layers.append(current_nodes)
                prev_nodes = current_nodes

        return graph, layers

    # 📌 Générer le graphe et les couches
    graph, layers = create_graph(actor)

    # 📌 Vérifier si les activations sont disponibles
    if activations is None or len(activations) == 0:
        activations = [[0.0 for _ in layer] for layer in layers]  # Valeurs neutres pour chaque couche

    # 📌 Positionner les neurones (centrés et ajustés)
    def position_nodes(graph, layers):
        positions = {}
        num_layers = len(layers)

        # Ajuster les espacements dynamiquement
        LAYER_SPACING = WIDTH // (num_layers + 1)
        max_neurons = max(len(layer) for layer in layers)
        NEURON_SPACING = HEIGHT // (max_neurons + 1)

        for layer_num, layer_nodes in enumerate(layers):
            x = LAYER_SPACING * (layer_num + 1)
            y_start = HEIGHT // 2 - (len(layer_nodes) * NEURON_SPACING) // 2

            for i, node in enumerate(layer_nodes):
                y = y_start + i * NEURON_SPACING
                positions[node] = (x, y)

        return positions

    positions = position_nodes(graph, layers)

    # 📌 Dessiner une connexion
    def draw_edge(screen, pos1, pos2, weight, highlight=False):
        color = GREEN if weight > 0 else RED
        width = int(1 + abs(weight) * 3) if highlight else 1
        pygame.draw.line(screen, color, pos1, pos2, width)

    # 📌 Dessiner les neurones avec activations
    def draw_neuron(screen, pos, activation=None):
        if activation is not None:
            # Si activation, colorier en rouge, sinon en gris
            color = RED if activation > 0.5 else GRAY
        else:
            color = BLUE
        pygame.draw.circle(screen, color, pos, NODE_RADIUS)
        pygame.draw.circle(screen, DARK_GRAY, pos, NODE_RADIUS, 2)  # Contour sombre

    # 📌 Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    # 📌 Effacer l'écran
    screen.fill(TRANSPARENT)

    # 📌 Dessiner les connexions
    for edge in graph.edges(data=True):
        src, dst, data = edge
        weight = data["weight"]
        highlight = True  # Toutes les connexions actives sont dessinées
        draw_edge(screen, positions[src], positions[dst], weight, highlight)

    # 📌 Dessiner les neurones
    for layer_num, layer_nodes in enumerate(layers):
        for i, node in enumerate(layer_nodes):
            activation = activations[layer_num][i] if layer_num < len(activations) and i < len(activations[layer_num]) else 0.0
            draw_neuron(screen, positions[node], activation)



    # 📌 Mise à jour de l'écran
    pygame.display.flip()
    clock.tick(30)  # 30 FPS

    return screen, clock,layers