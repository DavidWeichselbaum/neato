import math
from collections import defaultdict, deque
from pprint import pformat

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numba import njit


@njit
def act_linear(x): return x
@njit
def act_tanh(x): return math.tanh(x)
@njit
def act_sigmoid(x): return 1 / (1 + math.exp(-x))
@njit
def act_gaussian(x): return math.exp(-x**2)
@njit
def act_sine(x): return math.sin(x)
@njit
def act_square(x): return x**2
@njit
def act_softplus(x): return math.log1p(math.exp(x))

act_funcs = {
    'linear': act_linear,
    'tanh': act_tanh,
    'sigmoid': act_sigmoid,
    'gaussian': act_gaussian,
    'sine': act_sine,
    'square': act_square,
    'softplus': act_softplus,
}

@njit
def evaluate_network(conn_in, conn_out, conn_weight, node_func_ids, exec_order, input_indices, input_vals, output_indices):
    n_nodes = len(node_func_ids)
    n_conns = len(conn_out)
    values = np.zeros(n_nodes)

    # set inputs
    for i in range(len(input_indices)):
        values[input_indices[i]] = input_vals[i]

    # process nodes in topo order
    for k in range(len(exec_order)):
        idx = exec_order[k]

        # skip inputs
        is_input_node = False
        for ii in range(len(input_indices)):
            if idx == input_indices[ii]:
                is_input_node = True
                break
        if is_input_node:
            continue

        total = 0.0
        for j in range(n_conns):
            if conn_out[j] == idx:
                src = conn_in[j]
                total += values[src] * conn_weight[j]

        act_id = node_func_ids[idx]
        if act_id == 0:
            values[idx] = act_linear(total)
        elif act_id == 1:
            values[idx] = act_tanh(total)
        elif act_id == 2:
            values[idx] = act_sigmoid(total)
        elif act_id == 3:
            values[idx] = act_gaussian(total)
        elif act_id == 4:
            values[idx] = act_sine(total)
        elif act_id == 5:
            values[idx] = act_square(total)
        elif act_id == 6:
            values[idx] = act_softplus(total)
        else:
            values[idx] = 0.0

    return values[output_indices]


class Node:

    def __init__(self, node_id, is_input=False, is_output=False, is_bias=False, activation='linear'):
        self.id = node_id
        self.is_input = is_input
        self.is_output = is_output
        self.is_bias = is_bias
        self.activation_name = activation
        self.activation = act_funcs[activation]

        if self.is_bias:
            self.is_input = True
            self.activation = lambda x: 1
            self.activation_name = 'bias'

    def activate(self, input_):
        if isinstance(input_, np.ndarray):
            return np.array([self.activation(x) for x in input_])
        else:
            return self.activation(input_)

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)


class Connection:
    def __init__(self, in_node, out_node, weight):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)


class NEATNetwork:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections
        self.node_ids = [n.id for n in nodes]

        if self.detect_cycles(self.connections, self.node_ids):
            raise Exception("Network is not a directed acyclic graph")

        # for activation
        self.exec_order, self.layer_order = self.topo_sort()
        self.id_to_layer = {nid: layer_idx for layer_idx, layer in enumerate(self.layer_order) for nid in layer}
        self.node_index = {nid: i for i, nid in enumerate(self.node_ids)}
        self.conn_in = np.array([self.node_index[c.in_node] for c in self.connections])
        self.conn_out = np.array([self.node_index[c.out_node] for c in self.connections])
        self.conn_weight = np.array([c.weight for c in self.connections])
        self.node_map = {n.id: n for n in nodes}
        self.input_ids = [n.id for n in nodes if n.is_input]
        self.output_ids = [n.id for n in nodes if n.is_output]

        self.node_func_ids = np.array([
            {'linear': 0, 'tanh': 1, 'sigmoid': 2, 'gaussian': 3, 'sine': 4, 'square': 5, 'softplus': 6}.get(self.node_map[nid].activation_name, -1)
            for nid in self.node_ids
        ])

        self.exec_order_arr = np.array([self.node_index[nid] for nid in self.exec_order])
        self.input_idx_arr = np.array([self.node_index[nid] for nid in self.input_ids])
        self.output_idx_arr = np.array([self.node_index[nid] for nid in self.output_ids])

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def detect_cycles(self, connections, node_ids):
        adj = defaultdict(list)
        for c in connections:
            adj[c.in_node].append(c.out_node)

        visited = set()
        stack = set()

        def dfs(nid):
            if nid in stack:
                return True
            if nid in visited:
                return False
            visited.add(nid)
            stack.add(nid)
            for neighbor in adj[nid]:
                if dfs(neighbor):
                    return True
            stack.remove(nid)
            return False

        for nid in node_ids:
            if dfs(nid):
                return True
        return False

    def topo_sort(self):
        """ Kahn's algorithm for topological sorting"""
        adj = defaultdict(list)
        in_degree = {nid: 0 for nid in self.node_ids}
        for c in self.connections:
            adj[c.in_node].append(c.out_node)
            in_degree[c.out_node] += 1

        # Queue of nodes with no incoming edges (except inputs)
        q = deque([(nid, 0) for nid in self.node_ids if in_degree[nid] == 0])
        layers = defaultdict(list)
        topo_order = []
        while q:
            nid, level = q.popleft()
            layers[level].append(nid)
            topo_order.append(nid)
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    q.append((neighbor, level + 1))

        sorted_layers = [layers[i] for i in sorted(layers)]
        return topo_order, sorted_layers

    def activate(self, inputs):
        return evaluate_network(
            self.conn_in,
            self.conn_out,
            self.conn_weight,
            self.node_func_ids,
            self.exec_order_arr,
            self.input_idx_arr,
            np.array(inputs),
            self.output_idx_arr
        )

    def visualize(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.get_graph_plot(ax)
        plt.show()

    def get_graph_plot(self, ax):
        G = nx.DiGraph()
        node_labels = {}
        positions = {}
        node_colors = []

        for nid in self.node_ids:
            node = self.node_map[nid]
            G.add_node(nid)
            node_labels[nid] = str(nid)

            # Layout
            level = self.id_to_layer.get(nid, 0)
            level_ids = self.layer_order[level]
            n_nodes_level = len(level_ids)
            order_level = level_ids.index(nid) if nid in level_ids else 0
            position = (level, 1 - 1 / max(n_nodes_level, 1) * order_level)
            positions[nid] = position

            # Color
            if node.is_bias:
                node_colors.append('yellow')
            elif node.is_input:
                node_colors.append('green')
            elif node.is_output:
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgray')

        # Draw main graph
        edge_colors = []
        edge_labels = {}
        for c in self.connections:
            G.add_edge(c.in_node, c.out_node)
            edge_colors.append('red' if c.weight < 0 else 'blue')
            edge_labels[(c.in_node, c.out_node)] = f"{c.weight:.2f}"

        nx.draw(G, positions, ax=ax, labels=node_labels, with_labels=True,
                node_color=node_colors, edgecolors='black', node_size=600,
                edge_color=edge_colors, arrows=True, font_size=6)

        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels,
                                     font_size=6, font_color='black', ax=ax)

        # Inline activation plots
        for nid in self.node_ids:
            node = self.node_map[nid]
            if node.is_bias:
                continue  # skip plotting for bias node

            pos = positions[nid]
            x0, y0 = pos

            inset = ax.inset_axes([x0 - 0.05, y0 - 0.05, 0.1, 0.1], transform=ax.transData)
            xs = np.linspace(-5, 5, 50)
            ys = node.activate(xs)
            inset.plot(xs, ys, linewidth=1)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_facecolor('white')
            inset.set_frame_on(False)

        ax.axis('off')
