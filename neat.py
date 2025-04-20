from collections import defaultdict, deque
from pprint import pformat

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import networkx as nx


class Node:
    act_funcs = {
        'bias': lambda x: 1,
        'tanh': np.tanh,
        'relu': lambda x: np.maximum(0, x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'linear': lambda x: x,
    }

    def __init__(self, node_id, is_input=False, is_output=False, activation='tanh'):
        self.id = node_id
        self.is_input = is_input
        self.is_output = is_output
        self.activation = self.act_funcs[activation]

    def activate(self, input_):
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
                return True  # cycle detected
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
        values = np.zeros(len(self.node_ids))
        for i, val in enumerate(inputs):
            values[self.node_index[self.input_ids[i]]] = val

        for nid in self.exec_order:
            if nid in self.input_ids:
                continue
            idx = self.node_index[nid]
            incoming = self.conn_out == idx
            in_idxs = self.conn_in[incoming]
            weights = self.conn_weight[incoming]
            vals = values[in_idxs]
            total = np.sum(vals * weights)
            values[idx] = self.node_map[nid].activate(total)

        return [values[self.node_index[nid]] for nid in self.output_ids]

    def visualize(self):
        G = nx.DiGraph()

        node_labels = {}
        positions = {}
        for nid in self.node_ids:
            G.add_node(nid)
            node_labels[nid] = str(nid)

            level = self.id_to_layer[nid]
            level_ids = self.layer_order[level]
            n_nodes_level = len(level_ids)
            order_level = level_ids.index(nid)
            position = (level, 1 - 1/n_nodes_level * order_level)
            positions[nid] = position


        edge_colors = []
        edge_labels = {}
        for c in self.connections:
            G.add_edge(c.in_node, c.out_node)
            edge_colors.append('red' if c.weight < 0 else 'blue')
            edge_labels[(c.in_node, c.out_node)] = f"{c.weight:.2f}"

        nx.draw(G, positions, labels=node_labels, with_labels=True,
                node_color='lightgray', edgecolors='black', node_size=800,
                edge_color=edge_colors, arrows=True)
        nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels,
                                     font_size=8, font_color='black')

        plt.axis('off')
        plt.show()
