from copy import deepcopy
from random import random, choice, uniform
from collections import defaultdict, deque

import numpy as np

from NEAT.NEAT import Node, Connection, NEATNetwork, act_funcs


def random_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def prune_unused_nodes(nodes, connections, verbose=False):
    node_ids = [node.id for node in nodes]
    input_ids = [node.id for node in nodes if node.is_input]
    output_ids = [node.id for node in nodes if node.is_output]

    reachable = reachable_from_inputs(nodes, connections, input_ids)
    contributing = reaches_outputs(nodes, connections, output_ids)
    active_nodes = reachable & contributing

    post_pruning_nodes = [
        n for n in nodes if (
            n.id in active_nodes or
            n.is_input or
            n.is_output
        )
    ]
    post_pruning_ids = {n.id for n in post_pruning_nodes}
    if post_pruning_ids != set(node_ids):
        post_pruning_connections = [
            c for c in connections if c.in_node in post_pruning_ids and c.out_node in post_pruning_ids
        ]
        if verbose:
            print(f"🧹 Pruned {len(nodes) - len(post_pruning_nodes)} nodes and {len(connections) - len(post_pruning_connections)} connections")
        nodes = post_pruning_nodes
        connections = post_pruning_connections
    return nodes, connections


def create_random_net(
    n_inputs,
    n_outputs,
    n_hidden,
    n_connections,
    input_activations="linear",
    activation_choices=None,
    output_activations="linear",
    weight_range=(-2.0, 2.0),
    bias_range=(-1.0, 1.0),
    allow_recurrent=False,
    input_names=None,
    output_names=None,
):
    if input_names: assert len(input_names) == n_inputs
    if output_names: assert len(output_names) == n_outputs

    if activation_choices is None:
        activation_choices = list(act_funcs.keys())

    nodes = []
    node_id = 0

    # Input nodes
    input_nodes = []
    for i in range(n_inputs):
        if isinstance(input_activations, str):
            act = input_activations
        else:
            act = input_activations[i]

        name = input_names[i] if input_names else None
        nodes.append(Node(node_id, is_input=True, activation=act, name=name, bias=0.0))  # no bias for input
        input_nodes.append(node_id)
        node_id += 1

    # Hidden nodes
    hidden_nodes = []
    for _ in range(n_hidden):
        act = choice(activation_choices)
        bias = uniform(*bias_range)
        nodes.append(Node(node_id, activation=act, bias=bias))
        hidden_nodes.append(node_id)
        node_id += 1

    # Output nodes
    output_nodes = []
    for i in range(n_outputs):
        if isinstance(output_activations, str):
            act = output_activations
        else:
            act = output_activations[i]

        name = output_names[i] if output_names else None
        bias = uniform(*bias_range)
        nodes.append(Node(node_id, is_output=True, activation=act, name=name, bias=bias))
        output_nodes.append(node_id)
        node_id += 1

    all_nodes = input_nodes + hidden_nodes + output_nodes
    non_input_nodes = hidden_nodes + output_nodes

    connections = set()
    attempts = 0
    max_attempts = n_connections * 10

    while len(connections) < n_connections and attempts < max_attempts:
        src = choice(all_nodes)
        dst = choice(non_input_nodes)
        if not allow_recurrent and src >= dst:
            attempts += 1
            continue
        if (src, dst) not in connections and src != dst:
            weight = uniform(*weight_range)
            connections.add((src, dst, weight))
        attempts += 1

    conn_objs = [Connection(src, dst, w) for src, dst, w in connections]
    nodes, conn_objs = prune_unused_nodes(nodes, conn_objs)
    return NEATNetwork(nodes, conn_objs)


def creates_cycle(connections, src, dst):
    adj = defaultdict(list)
    for c in connections:
        adj[c.in_node].append(c.out_node)

    # Simulate the new edge
    adj[src].append(dst)

    visited = set()
    stack = deque([dst])
    while stack:
        node = stack.pop()
        if node == src:
            return True  # Cycle detected
        if node not in visited:
            visited.add(node)
            stack.extend(adj[node])
    return False


def reachable_from_inputs(nodes, connections, input_ids):
    adj = defaultdict(list)
    for c in connections:
        adj[c.in_node].append(c.out_node)
    visited = set(input_ids)
    stack = list(input_ids)
    while stack:
        n = stack.pop()
        for neighbor in adj[n]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


def reaches_outputs(nodes, connections, output_ids):
    rev_adj = defaultdict(list)
    for c in connections:
        rev_adj[c.out_node].append(c.in_node)
    visited = set(output_ids)
    stack = list(output_ids)
    while stack:
        n = stack.pop()
        for neighbor in rev_adj[n]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return visited


def duplicate_subnet_with_scaled_outputs(nodes, connections, input_ids, output_ids, node_pick_prob=0.5, max_subnet_size=-1, verbose=False):
    node_map = {n.id: n for n in nodes}
    con_map = {}  # maps from a node to its outgoing connections
    for c in connections:
        con_map.setdefault(c.in_node, []).append(c)

    # Pick a random non-input/output node as seed
    hidden_ids = [n.id for n in nodes if n.id not in input_ids + output_ids]
    if not hidden_ids:
        return nodes, connections

    seed_id = choice(hidden_ids)
    subnet_ids = set([seed_id])
    frontier = [seed_id]

    # Recursively grow the subnet downstream
    while frontier:
        current = frontier.pop()
        for c in con_map.get(current, []):
            if max_subnet_size > 0 and len(subnet_ids) >= max_subnet_size:
                break

            if c.out_node in output_ids:
                continue

            if c.out_node not in subnet_ids and random() < node_pick_prob:
                subnet_ids.add(c.out_node)
                frontier.append(c.out_node)

    if not subnet_ids:
        return nodes, connections

    if verbose:
        print(f"🔀 Selected subnet for duplication: {sorted(subnet_ids)}")

    # Duplicate nodes
    max_node_id = max(n.id for n in nodes)
    id_map = {}
    new_nodes = []
    for nid in subnet_ids:
        max_node_id += 1
        new_id = max_node_id
        id_map[nid] = new_id

        orig = node_map[nid]
        new_nodes.append(
            Node(new_id, activation=orig.activation_name, bias=orig.bias)
        )
        if verbose:
            print(f"🆕 Duplicated node {nid} -> {new_id}")

    new_connections = []
    for c in connections:
        # Duplicate intra-subnet connections
        if c.in_node in subnet_ids and c.out_node in subnet_ids:
            new_connections.append(
                Connection(id_map[c.in_node], id_map[c.out_node], c.weight)
            )
        # Duplicate input connections into the subnet
        elif c.out_node in subnet_ids and c.in_node not in subnet_ids:
            new_connections.append(
                Connection(c.in_node, id_map[c.out_node], c.weight)
            )
        # Duplicate and scale output connections
        elif c.in_node in subnet_ids and c.out_node not in subnet_ids:
            halved_weight = c.weight * 0.5  # half the outgoing weight to maintain the same activations of the whole net
            c.weight = halved_weight  # also adapt the orginal connection
            new_connections.append(
                Connection(id_map[c.in_node], c.out_node, halved_weight)
            )
        else:
            continue

    if verbose:
        print(f"Add a subnet of {len(new_nodes)} nodes and {len(new_connections)} connections")

    return nodes + new_nodes, connections + new_connections


def mutate_net(
    net,

    global_mutation_prob=0.5,

    mutate_weight_prob=0.9,
    perturb_prob=0.8,
    perturb_std=0.1,
    reset_weight_prob=0.1,
    weight_range=(-2.0, 2.0),

    mutate_bias_prob=0.2,
    perturb_bias_std=0.05,
    reset_bias_prob=0.05,
    bias_range=(-1.0, 1.0),

    activation_change_prob=0.1,

    add_conn_prob=0.1,
    del_conn_prob=0.1,
    add_node_prob=0.05,
    del_node_prob=0.05,
    add_conn_attempts=30,

    duplication_prob=0.05,
    node_pick_prob=0.5,
    max_subnet_size=-1,

    activation_choices=None,
    allow_recurrent=False,
    prune_unused=True,
    verbose=False
):

    if random() < global_mutation_prob:
        return net.copy()

    nodes = deepcopy(net.nodes)
    connections = deepcopy(net.connections)

    if activation_choices is None:
        activation_choices = list(act_funcs.keys())

    node_ids = [n.id for n in nodes]
    max_node_id = max(node_ids) + 1
    con_set = {(c.in_node, c.out_node) for c in connections}
    node_map = {n.id: n for n in nodes}

    input_ids = [n.id for n in nodes if n.is_input]
    output_ids = [n.id for n in nodes if n.is_output]
    hidden_ids = [nid for nid in node_ids if nid not in input_ids and nid not in output_ids]
    non_input_ids = hidden_ids + output_ids

    mutate_connection_weights(connections, mutate_weight_prob, perturb_prob, perturb_std,
                              reset_weight_prob, weight_range, verbose)

    mutate_node_biases(nodes, mutate_bias_prob, perturb_prob, perturb_bias_std,
                       reset_bias_prob, bias_range, verbose)

    mutate_activations(nodes, activation_change_prob, activation_choices, verbose)

    add_random_connection(connections, node_ids, non_input_ids, con_set, add_conn_prob,
                          allow_recurrent, perturb_bias_std, verbose)

    delete_random_connection(connections, node_ids, del_conn_prob, verbose)

    nodes, connections, max_node_id = add_node_split(nodes, connections, node_ids, add_node_prob,
                                                     activation_choices, bias_range,
                                                     max_node_id, verbose)

    nodes, connections = delete_hidden_node(nodes, connections, hidden_ids,
                                            del_node_prob, verbose)

    if random() < duplication_prob:
        nodes, connections = duplicate_subnet_with_scaled_outputs(
            nodes, connections, input_ids, output_ids,
            node_pick_prob=node_pick_prob,
            max_subnet_size=max_subnet_size, verbose=verbose
        )

    if prune_unused:
        nodes, connections = prune_unused_nodes(nodes, connections)

    return NEATNetwork(nodes, connections)


def mutate_connection_weights(connections, mutate_prob, perturb_prob, std, reset_prob, weight_range, verbose):
    for c in connections:  # scale wit number of connections
        if random() < mutate_prob:
            if random() < perturb_prob:
                old = c.weight
                c.weight += np.random.normal(0, std)
                if verbose: print(f"🔧 Nudged weight {old:.3f} -> {c.weight:.3f} for conn {c.in_node}->{c.out_node}")
            if random() < reset_prob:
                old = c.weight
                c.weight = uniform(*weight_range)
                if verbose: print(f"🎲 Reset weight {old:.3f} -> {c.weight:.3f} for conn {c.in_node}->{c.out_node}")


def mutate_node_biases(nodes, mutate_prob, perturb_prob, std, reset_prob, bias_range, verbose):
    for n in nodes:  # scale with number of nodes
        if not n.is_input and random() < mutate_prob:
            if random() < perturb_prob:
                old = n.bias
                n.bias += np.random.normal(0, std)
                if verbose: print(f"⚙️ Nudged bias {old:.3f} -> {n.bias:.3f} on node {n.id}")
            if random() < reset_prob:
                old = n.bias
                n.bias = uniform(*bias_range)
                if verbose: print(f"🎯 Reset bias {old:.3f} -> {n.bias:.3f} on node {n.id}")


def mutate_activations(nodes, change_prob, activation_choices, verbose):
    for n in nodes:  # scale with number of nodes
        if not n.is_input and not n.is_output and random() < change_prob:
            old_act = n.activation_name
            new_act = choice(activation_choices)
            n.activation = act_funcs[new_act]
            n.activation_name = new_act
            if verbose: print(f"⚡ Changed activation of node {n.id} from {old_act} to {new_act}")


def add_random_connection(connections, node_ids, non_input_ids, con_set,
                          add_prob, allow_recurrent, std, verbose):
    for src in node_ids:  # scale with number of nodes
        if random() < add_prob:
            dst = choice(non_input_ids)
            if src == dst or (src, dst) in con_set:
                continue
            if allow_recurrent or not creates_cycle(connections, src, dst):
                w = np.random.normal(0, std)
                connections.append(Connection(src, dst, w))
                con_set.add((src, dst))
                if verbose:
                    print(f"➕ Added connection {src} -> {dst} with weight {w:.3f}")
                break


def delete_random_connection(connections, node_ids, prob, verbose):
    for nid in node_ids:  # scale with number of nodes
        if random() < prob:
            conns = [c for c in connections if c.in_node == nid or c.out_node == nid]
            if conns:
                c = choice(conns)
                connections.remove(c)
                if verbose: print(f"❌ Removed connection {c.in_node} -> {c.out_node}")
                break


def add_node_split(nodes, connections, node_ids, prob, activation_choices, bias_range, max_node_id, verbose):
    for nid in node_ids:  # scale with number of nodes
        if random() < prob and connections:
            c = choice(connections)
            connections.remove(c)
            new_act = choice(activation_choices)
            new_bias = uniform(*bias_range)
            new_node = Node(max_node_id, activation=new_act, bias=new_bias)
            nodes.append(new_node)
            connections.append(Connection(c.in_node, new_node.id, 1.0))
            connections.append(Connection(new_node.id, c.out_node, c.weight))
            if verbose: print(f"✨ Split connection {c.in_node}->{c.out_node} with new node {new_node.id} ({new_act}) bias={new_bias:.3f}")
            max_node_id += 1
            break
    return nodes, connections, max_node_id


def delete_hidden_node(nodes, connections, hidden_ids, prob, verbose):
    for nid in hidden_ids:  # scale with number of hidden nodes
        if random() < prob:
            nodes = [n for n in nodes if n.id != nid]
            removed_conns = [c for c in connections if c.in_node == nid or c.out_node == nid]
            connections = [c for c in connections if c.in_node != nid and c.out_node != nid]
            if verbose: print(f"🗑️ Removed node {nid} and {len(removed_conns)} connected links")
            break
    return nodes, connections
