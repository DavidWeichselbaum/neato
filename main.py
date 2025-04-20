import random

import matplotlib.pyplot as plt

from neat import Node, Connection, NEATNetwork
from utils.utils import render_cpp_image


def create_random_neat(n_inputs, n_outputs, n_hidden, n_connections,
                       activation_choices=None, weight_range=(-2.0, 2.0), allow_recurrent=False):
    if activation_choices is None:
        activation_choices = list(Node.act_funcs.keys())

    nodes = []
    node_id = 0

    # Bias node
    nodes.append(Node(node_id, is_bias=True))
    bias_id = node_id
    node_id += 1

    # Input nodes
    input_nodes = []
    for _ in range(n_inputs):
        nodes.append(Node(node_id, is_input=True))
        input_nodes.append(node_id)
        node_id += 1

    # Hidden nodes
    hidden_nodes = []
    for _ in range(n_hidden):
        act = random.choice(activation_choices)
        nodes.append(Node(node_id, activation=act))
        hidden_nodes.append(node_id)
        node_id += 1

    # Output nodes
    output_nodes = []
    for _ in range(n_outputs):
        act = random.choice(activation_choices)
        nodes.append(Node(node_id, is_output=True, activation=act))
        output_nodes.append(node_id)
        node_id += 1

    all_nodes = [bias_id] + input_nodes + hidden_nodes + output_nodes
    non_input_nodes = hidden_nodes + output_nodes

    # Generate connections
    connections = set()
    attempts = 0
    max_attempts = n_connections * 10  # prevent infinite loops

    while len(connections) < n_connections and attempts < max_attempts:
        src = random.choice(all_nodes)
        dst = random.choice(non_input_nodes)

        # Optional: disallow cycles if not allowing recurrent
        if not allow_recurrent and src >= dst:
            attempts += 1
            continue

        if (src, dst) not in connections and src != dst:
            weight = random.uniform(*weight_range)
            connections.add((src, dst, weight))

        attempts += 1

    conn_objs = [Connection(src, dst, w) for src, dst, w in connections]
    return NEATNetwork(nodes, conn_objs)


while True:
    net = create_random_neat(2, 3, 16, 62)
    net.visualize()

    image = render_cpp_image(net)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
