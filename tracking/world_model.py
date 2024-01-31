import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from .tracker import Tracker


class WorldModel:
    def __init__(self, n_init: int = 1, w: float = 0.5, gating_min=0.4, gating_max=0.8):
        # Multi object tracking initialization
        self.tracker = Tracker(max_age=1e6, n_init=n_init, w=w, gating_min=gating_min, gating_max=gating_max)

        self.colours = []
        for _i in range(1000):
            color = np.random.rand(3)
            self.colours.append(color.tolist())

    def run(self, det_list):
        # Run tracker
        self.tracker.predict(np.eye(3, 3) * 0.001)
        self.tracker.update(det_list)


def isotropic_unit_vectors():
    # Note: we must use arccos in the definition of theta to prevent bunching of points toward the poles
    phi = np.linspace(0, 2 * np.pi, 5000)
    theta = np.linspace(0, np.pi, 5000)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def rgb_to_plotly_color(rgb_color):
    return f"rgb({int(rgb_color[0] * 255)}, {int(rgb_color[1] * 255)}, {int(rgb_color[2] * 255)})"


def get_colors_ids():
    # Create a list of 100 unique IDs from 0 to 99
    unique_ids = list(range(100))

    # Generate an array of equally spaced values from 0 to 1
    colors = np.linspace(0, 1, len(unique_ids))

    # Create a colormap using the 'viridis' color map (or any other available color map)
    colormap = plt.get_cmap("viridis")

    # Map the colors array to actual colors using the colormap and convert them to Plotly format
    unique_colors = [rgb_to_plotly_color(colormap(color)) for color in colors]

    # Create a dictionary to map each unique ID to a color
    return dict(zip(unique_ids, unique_colors))


def plotly_sphere(center, radius, color):
    u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : np.pi : 50j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)

    return go.Mesh3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), alphahull=0, color=color, opacity=0.5)
