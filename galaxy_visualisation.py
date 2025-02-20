from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

class GalaxyVisualizer:
    @staticmethod
    def plot_3d_cluster(galaxies):
        """Plot a 3D scatter plot of galaxy positions."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection='3d')
        colors = np.random.rand(len(galaxies))
        for galaxy, color in zip(galaxies, colors):
            ax.scatter(galaxy.x, galaxy.y, galaxy.z, c=color, s=0.05, alpha=0.6)
        ax.set_title('3D Spatial Plot of Galaxy Positions')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        plt.show()

    @staticmethod
    def plot_contour(galaxies, z_min, z_max, bins=1000):
        """Plot a 2D slice and contour plot of galaxy positions."""
        x_slice = [galaxy.x for galaxy in galaxies if z_min <= galaxy.z <= z_max]
        y_slice = [galaxy.y for galaxy in galaxies if z_min <= galaxy.z <= z_max]

        hist, xedges, yedges = np.histogram2d(x_slice, y_slice, bins=bins)
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        y_centers = (yedges[:-1] + yedges[1:]) / 2

        plt.figure(figsize=(8, 6))
        plt.scatter(x_slice, y_slice, s=5, alpha=0.7)
        plt.title(f'2D Slice of 3D Data (Z in [{z_min}, {z_max}])')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.contourf(x_centers, y_centers, hist.T, levels=400, cmap='inferno')
        plt.colorbar(label='Projected Density')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()