import matplotlib.pyplot as plt

from neat import Node, Connection, NEATNetwork
from utils.utils import render_cpp_image


nodes = [
    Node(0, is_input=True, activation='bias'),

    Node(1, is_input=True),
    Node(2, is_input=True),

    Node(3, is_output=True, activation='sigmoid'),
    Node(4, is_output=True, activation='sigmoid'),
    Node(5, is_output=True, activation='sigmoid'),

    Node(6, activation='relu'),
    Node(7, activation='sigmoid'),
    Node(8, activation='tanh'),
]

connections = [
    Connection(1, 6, 0.5),
    Connection(2, 6, 0.1),

    Connection(6, 7, -0.1),
    Connection(1, 7, 1),

    Connection(7, 8, 2),

    Connection(1, 3, 2),
    Connection(8, 3, 1),
    Connection(8, 4, 1),
    Connection(8, 5, 1),
    Connection(0, 5, 1),
]

net = NEATNetwork(nodes, connections)
net.visualize()
out = net.activate([0.5, 0.5, 0.5])
print(out)

image = render_cpp_image(net)
plt.imshow(image)
plt.axis('off')
plt.show()
