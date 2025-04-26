import numpy as np


def render_cppn_image(net, width=128, height=128, input_range=(-1, 1), scale=1.0):
    """
    Renders a 2D RGB image using the NEAT network.
    Inputs are (x, y), outputs are interpreted as RGB in [0, 1]
    """
    image = np.zeros((height, width, 3))

    xs = np.linspace(input_range[0], input_range[1], width)
    ys = np.linspace(input_range[0], input_range[1], height)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            inputs = [x * scale, y * scale]
            out = net.activate(inputs)
            rgb = np.clip(out, 0, 1)
            if len(rgb) == 1:
                rgb = [rgb[0]] * 3  # grayscale
            elif len(rgb) >= 3:
                rgb = rgb[:3]
            image[i, j] = rgb

    return image
