import matplotlib.pyplot as plt
import numpy as np


def surface_plot(
    X: list | np.ndarray,
    Y: list | np.ndarray,
    Z: list | np.ndarray,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    azim: int = 225,
    elev: int = 25,
    cmap: str = 'plasma',
) -> plt.Figure:
    """
    Create a 3D surface plot from the given input data.

    Parameters
    ----------
    X : list or numpy.ndarray
        1D array or list of values for the x-axis.
    Y : list or numpy.ndarray
        1D array or list of values for the y-axis.
    Z : list or numpy.ndarray
        1D array or list of values for the z-axis. Must be reshapeable to match the
        grid dimensions.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    zlabel : str
        Label for the z-axis.
    title : str
        Title of the plot.
    azim : int, optional
        Azimuthal angle for 3D view initialization (default is 225).
    elev : int, optional
        Elevation angle for 3D view initialization (default is 25).
    cmap : str, optional
        Colormap for the surface plot (default is 'plasma').

    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the generated 3D surface plot.

    Notes
    -----
    - Ensure that `Z` is reshapeable into the meshgrid shape created by `X` and `Y`.
    - Use the `azim` and `elev` parameters to adjust the initial view of the 3D plot.
    """
    X_arr, Y_arr, Z_arr = np.array(X), np.array(Y), np.array(Z)

    X_grid, Y_grid = np.meshgrid(X_arr, Y_arr)
    Z_grid = Z_arr.reshape(X_grid.shape).T

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap, alpha=0.5)

    cbar = fig.colorbar(surface, shrink=0.5, aspect=10)
    cbar.set_label(zlabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax.yaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))
    ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 1.0))

    ax.grid(True)

    return fig
