import matplotlib.pyplot as plt
import numpy as np

from utils.utils import render_cpp_image
from utils.evolution import create_random_net, mutate_net


class CCPEvolution:
    def __init__(self, base_net, img_size=128):
        self.base_net = base_net
        self.img_size = img_size
        self.grid_shape = (3, 3)

        # Image window
        self.fig_img, self.axes_img = plt.subplots(*self.grid_shape, figsize=(8, 8))
        self.fig_img.canvas.manager.set_window_title("CPP Image Grid")

        # Network window
        self.fig_net, self.axes_net = plt.subplots(*self.grid_shape, figsize=(8, 8))
        self.fig_net.canvas.manager.set_window_title("Network Grid")

        self.nets = [[None for _ in range(3)] for _ in range(3)]
        self.images = [[None for _ in range(3)] for _ in range(3)]

        self.generate_grid()
        self.fig_img.canvas.mpl_connect('button_press_event', self.on_click)
        plt.tight_layout()
        plt.show()

    def generate_grid(self):
        self.fig_img.suptitle("Click a CPP to evolve", fontsize=10)
        self.fig_net.suptitle("Corresponding Network Topologies", fontsize=10)

        for i in range(3):
            for j in range(3):
                print(f"Net {i},{j}")
                if i == 0 and j == 0:  # First is ancenstor
                    self.nets[i][j] = self.base_net
                else:
                    self.nets[i][j] = mutate_net(self.base_net)

                # CPP image
                img = render_cpp_image(self.nets[i][j], width=self.img_size, height=self.img_size)
                self.images[i][j] = img
                self.axes_img[i][j].imshow(img)
                self.axes_img[i][j].axis('off')
                self.axes_img[i][j].set_title(f"{i},{j}", fontsize=8)

                # Clear net plot
                self.axes_net[i][j].cla()

                # Render NEAT network to axis
                self.nets[i][j].get_graph_plot(self.axes_net[i][j])
                self.axes_net[i][j].set_title(f"{i},{j}", fontsize=8)
                self.axes_net[i][j].axis('off')

        self.fig_img.canvas.draw()
        self.fig_net.canvas.draw()

    def on_click(self, event):
        if event.inaxes is None:
            return

        # Find clicked image
        for i in range(3):
            for j in range(3):
                if self.axes_img[i][j] == event.inaxes:
                    print(f"🖱️ Clicked on {i},{j}")
                    self.base_net = self.nets[i][j]
                    self.generate_grid()
                    return


net = create_random_net(2, 3, 16, 64)
# print(net)
# net.visualize()
CCPEvolution(net, img_size=64)
