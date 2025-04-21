import matplotlib.pyplot as plt

from utils.utils import render_cpp_image
from utils.evolution import create_random_neat


while True:
    net = create_random_neat(2, 3, 16, 62)
    net.visualize()

    image = render_cpp_image(net)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
