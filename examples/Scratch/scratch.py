import numpy as np
import pyvista as pv
import matplotlib.cm as mpl_cm


def plot_temperature_field(plotter, nodes, T, dims, cmap="inferno", opacity=0.75):
    """
    Plot the temperature field on a structured grid.

    Parameters:
      plotter: PyVista plotter instance.
      nodes: NumPy array of shape (n_nodes, 3) with node coordinates.
      T: NumPy array of nodal temperatures.
      dims: Tuple (nx+1, ny+1, nz+1) defining the grid dimensions.
      cmap: Colormap name for temperature.
      opacity: Global opacity for the temperature mesh.
    """
    # Reshape nodes into structured grid.
    grid_points = nodes.reshape(dims + (3,))
    grid = pv.StructuredGrid()
    grid.points = grid_points.reshape(-1, 3)
    grid.dimensions = dims
    grid["Temperature"] = T
    plotter.add_mesh(grid, scalars="Temperature", cmap=cmap, opacity=opacity, show_edges=True)


def plot_density_field(plotter, nodes, density, dims, cmap="viridis", opacity=1.0):
    """
    Plot the density field on a structured grid as an overlay.

    Parameters:
      plotter: PyVista plotter instance.
      nodes: NumPy array of shape (n_nodes, 3).
      density: NumPy array of nodal density values (should be same length as nodes).
               (If density is element-centered, convert it to nodal first.)
      dims: Tuple (nx+1, ny+1, nz+1) defining the grid dimensions.
      cmap: Colormap name for density.
      opacity: Opacity for the density overlay.
    """
    # Reshape nodes into structured grid.
    grid_points = nodes.reshape(dims + (3,))
    grid = pv.StructuredGrid()
    grid.points = grid_points.reshape(-1, 3)
    grid.dimensions = dims

    # Add the density field.
    grid["Density"] = density.flatten()  # Make sure density is nodal (same number of points)

    # Here we assume the density values will be mapped by the colormap.
    # For instance, low density might map to a light color (or even transparent)
    # and high density to a darker, more opaque color.
    plotter.add_mesh(grid, scalars="Density", cmap=cmap, opacity=opacity, show_edges=False)


# Example usage:
if __name__ == '__main__':
    # For example, create a simple structured grid.
    nx, ny, nz = 10, 10, 1  # 2D grid extruded once (for simplicity)
    dims = (nx + 1, ny + 1, nz + 1)
    # Create nodes: a uniform grid over [0, 1] in x and y, and [0, 1] in z.
    x = np.linspace(0, 1, dims[0])
    y = np.linspace(0, 1, dims[1])
    z = np.linspace(0, 1, dims[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    nodes = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Create a synthetic temperature field (for instance, a gradient in the y direction).
    T = Y.ravel() * 100 + 300  # Temperature goes from 300 to 400 K.

    # Create a synthetic density field.
    # Let's say the "object" occupies the central region.
    nodal_density = np.ones(dims)
    # Set low density (air) elsewhere.
    nodal_density[:2, :, :] = 1e-3
    nodal_density[-2:, :, :] = 1e-3
    nodal_density[:, :2, :] = 1e-3
    nodal_density[:, -2:, :] = 1e-3

    # Create a PyVista plotter.
    plotter = pv.Plotter(shape=(1, 1), window_size=(800, 600))

    # Plot the temperature field.
    plot_temperature_field(plotter, nodes, T, dims, cmap="inferno", opacity=0.75)

    # Plot the density overlay.
    # We can use a colormap like "viridis" where the density is represented by color.
    # Optionally, you can adjust the opacity of the density overlay to allow the temperature
    # colors to be visible underneath.
    plot_density_field(plotter, nodes, nodal_density, dims, cmap="viridis", opacity=0.8)

    # Optionally, if you want to annotate with load or heat generation information,
    # you can add extra text or legend entries.
    plotter.add_text("Temperature (colored by inferno) with Density Overlay", position="upper_left")

    plotter.show()