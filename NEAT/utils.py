import numpy as np
from copy import deepcopy
from random import random, choice, uniform

from NEAT.NEAT import Node, Connection, NEATNetwork, act_funcs


def create_random_net(
    n_inputs, n_outputs, n_hidden, n_connections,
    input_activation_choices=["linear"], activation_choices=None, output_activation_choices=["linear"],
    weight_range=(-2.0, 2.0), allow_recurrent=False,
    input_names=None, output_names=None,
):
    if input_names: assert len(input_names) == n_inputs
    if output_names: assert len(output_names) == n_outputs

    if activation_choices is None:
        activation_choices = list(act_funcs.keys())

    nodes = []
    node_id = 0

    # Bias node
    nodes.append(Node(node_id, is_bias=True))
    bias_id = node_id
    node_id += 1

    # Input nodes
    input_nodes = []
    for i in range(n_inputs):
        act = choice(input_activation_choices)
        name = None
        if input_names:
            name = input_names[i]
        nodes.append(Node(node_id, is_input=True, activation=act, name=name))
        input_nodes.append(node_id)
        node_id += 1

    # Hidden nodes
    hidden_nodes = []
    for _ in range(n_hidden):
        act = choice(activation_choices)
        nodes.append(Node(node_id, activation=act))
        hidden_nodes.append(node_id)
        node_id += 1

    # Output nodes
    output_nodes = []
    for i in range(n_outputs):
        act = choice(output_activation_choices)
        name = None
        if output_names:
            name = output_names[i]
        nodes.append(Node(node_id, is_output=True, activation=act, name=name))
        output_nodes.append(node_id)
        node_id += 1

    all_nodes = [bias_id] + input_nodes + hidden_nodes + output_nodes
    non_input_nodes = hidden_nodes + output_nodes

    # Generate connections
    connections = set()
    attempts = 0
    max_attempts = n_connections * 10  # prevent infinite loops

    while len(connections) < n_connections and attempts < max_attempts:
        src = choice(all_nodes)
        dst = choice(non_input_nodes)

        # Optional: disallow cycles if not allowing recurrent
        if not allow_recurrent and src >= dst:
            attempts += 1
            continue

        if (src, dst) not in connections and src != dst:
            weight = uniform(*weight_range)
            connections.add((src, dst, weight))

        attempts += 1

    conn_objs = [Connection(src, dst, w) for src, dst, w in connections]
    return NEATNetwork(nodes, conn_objs)


def mutate_net(
    net,

    global_mutation_prob=5.0,               # Global chance to apply any mutation at all

    mutate_weight_prob=0.9,                 # Chance to mutate weights
    perturb_prob=0.8,                       # Chance to nudge vs reinitialize weight
    perturb_std=0.1,                        # Std dev for Gaussian perturbation
    reset_weight_prob=0.1,                  # Chance to reset weight instead of nudging
    weight_range=(-2.0, 2.0),               # Bounds for random weights

    mutate_activation_prob=0.1,             # Chance to change a node's activation function
    activation_change_prob=0.1,             # Chance to mutate each individual node

    add_conn_prob=0.2,                      # Probability to add a connection
    del_conn_prob=0.2,                      # Probability to delete a connection
    add_node_prob=0.1,                      # Probability to add a node
    del_node_prob=0.1,                      # Probability to delete a hidden node
    add_conn_attempts=30,                   # Attempts to find a valid new connection

    activation_choices=None,
    allow_recurrent=False,
    verbose=False,
):
    if random() > global_mutation_prob:
        return net.copy()

    if activation_choices is None:
        activation_choices = list(act_funcs.keys())

    nodes = deepcopy(net.nodes)
    connections = deepcopy(net.connections)

    node_ids = [n.id for n in nodes]
    max_node_id = max(node_ids) + 1
    con_set = {(c.in_node, c.out_node) for c in connections}
    node_map = {n.id: n for n in nodes}

    input_ids = [n.id for n in nodes if n.is_input or n.is_bias]
    output_ids = [n.id for n in nodes if n.is_output]
    hidden_ids = [nid for nid in node_ids if nid not in input_ids and nid not in output_ids]
    non_input_ids = hidden_ids + output_ids

    # Mutate weights
    if random() < mutate_weight_prob:
        for c in connections:
            if random() < perturb_prob:
                old = c.weight
                c.weight += np.random.normal(0, perturb_std)
                if verbose: print(f"🔧 Nudged weight {old:.3f} -> {c.weight:.3f} for conn {c.in_node}->{c.out_node}")
            if random() < reset_weight_prob:
                old = c.weight
                c.weight = uniform(*weight_range)
                if verbose: print(f"🎲 Reset weight {old:.3f} -> {c.weight:.3f} for conn {c.in_node}->{c.out_node}")

    # Mutate activation functions
    if random() < mutate_activation_prob:
        for n in nodes:
            if not (n.is_input or n.is_bias or n.is_output):
                if random() < activation_change_prob:
                    old_act = [k for k, v in act_funcs.items() if v == n.activation]
                    new_act = choice(activation_choices)
                    n.activation = act_funcs[new_act]
                    if verbose: print(f"⚡ Changed activation of node {n.id} from {old_act[0] if old_act else 'unknown'} to {new_act}")

    # Add connection
    if random() < add_conn_prob:
        for _ in range(add_conn_attempts):
            src = choice(node_ids)
            dst = choice(non_input_ids)
            if src != dst and (src, dst) not in con_set:
                if allow_recurrent or src < dst:
                    w = uniform(*weight_range)
                    connections.append(Connection(src, dst, w))
                    con_set.add((src, dst))
                    if verbose: print(f"➕ Added connection {src} -> {dst} with weight {w:.3f}")
                    break

    # Delete random connection
    if connections and random() < del_conn_prob:
        c = choice(connections)
        connections.remove(c)
        if verbose: print(f"❌ Removed connection {c.in_node} -> {c.out_node}")

    # Add node
    if random() < add_node_prob and connections:
        c = choice(connections)
        connections.remove(c)
        new_act = choice(activation_choices)
        new_node = Node(max_node_id, activation=new_act)
        nodes.append(new_node)
        connections.append(Connection(c.in_node, new_node.id, 1.0))
        connections.append(Connection(new_node.id, c.out_node, c.weight))
        if verbose: print(f"✨ Split connection {c.in_node}->{c.out_node} with new node {new_node.id} ({new_act})")
        max_node_id += 1

    # Delete hidden node
    if hidden_ids and random() < del_node_prob:
        nid = choice(hidden_ids)
        nodes = [n for n in nodes if n.id != nid]
        removed_conns = [c for c in connections if c.in_node == nid or c.out_node == nid]
        connections = [c for c in connections if c.in_node != nid and c.out_node != nid]
        if verbose: print(f"🗑️ Removed node {nid} and {len(removed_conns)} connected links")

    return NEATNetwork(nodes, connections)
