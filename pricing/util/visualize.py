import matplotlib.pyplot as plt
import numpy as np

from pricing.static.optimize import GradientDescent
from matplotlib.animation import FuncAnimation


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
    cmap: str = "plasma",
    alpha: float = 0.9,
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
    alpha : float, optional
        Alpha value (transparency) of the plot
        default: 0.9

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
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap=cmap, alpha=alpha)

    #cbar = fig.colorbar(surface, shrink=0.5, aspect=10)
    #cbar.set_label(zlabel)

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


def line_plot(
    X: list | np.ndarray, Y: list | np.ndarray, xlabel: str, ylabel: str, title: str
):
    """
    Create a 2D line plot from the given input data.

    Parameters
    ----------
    X : list or numpy.ndarray
        1D array or list of values for the x-axis.
    Y : list or numpy.ndarray
        1D array or list of values for the y-axis.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the generated 2D line plot.
    """
    fig, ax = plt.subplots()
    ax.plot(X, Y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig


def plot_descent_two_tiers(
    X: list | np.ndarray,
    Y: list | np.ndarray,
    Z: list | np.ndarray,
    descent: GradientDescent,
    title: str,
    azim: int = 225,
    elev: int = 25,
    cmap: str = "plasma",
) -> plt.Figure:

    x = [price[0] for price in descent.price_history]
    y = [price[1] for price in descent.price_history]
    z = descent.profit_history

    fig = surface_plot(
        X,
        Y,
        Z,
        "Tier 1 Price",
        "Tier 2 Price",
        "Expected Profit / Customer",
        title=title,
        elev=15,
        alpha=0.5,
    )

    ax = fig.axes[0]

    ax.plot(x, y, z, color="green", marker="o", linewidth="0.5", markersize=0.7)

    ax.plot(x[-1], y[-1], z[-1], "ro")

    return fig


def compare_descents_two_tiers(
    X: list | np.ndarray,
    Y: list | np.ndarray,
    Z: list | np.ndarray,
    descent1: GradientDescent,
    descent2: GradientDescent,
    title: str,
    azim: int = 225,
    elev: int = 25,
    cmap: str = "plasma",
) -> plt.Figure:

    x1 = [price[0] for price in descent1.price_history]
    y1 = [price[1] for price in descent1.price_history]
    z1 = descent1.profit_history
    x2 = [price[0] for price in descent2.price_history]
    y2 = [price[1] for price in descent2.price_history]
    z2 = descent2.profit_history

    fig = surface_plot(
        X,
        Y,
        Z,
        "Tier 1 Price",
        "Tier 2 Price",
        "Expected Profit / Customer",
        title=title,
        elev=15,
        alpha=0.5,
    )

    ax = fig.axes[0]

    ax.plot(x1, y1, z1, color="green", marker="o", linewidth="0.5", markersize=0.7)

    ax.plot(x2, y2, z2, color="blue", marker="o", linewidth="0.5", markersize=0.7)

    ax.plot(x1[-1], y1[-1], z1[-1], "ro")
    ax.plot(x2[-1], y2[-1], z2[-1], "ro")

    return fig


def compare_n_descents_two_tiers(
    X: list | np.ndarray,
    Y: list | np.ndarray,
    Z: list | np.ndarray,
    descents: list[GradientDescent],
    labels: list[str],
    title: str,
    azim: int = 225,
    elev: int = 25,
    cmap: str = "plasma",
) -> plt.Figure:

    fig = surface_plot(
        X,
        Y,
        Z,
        "Tier 1 Price",
        "Tier 2 Price",
        "Expected Profit / Customer",
        title=title,
        elev=15,
        alpha=0.5,
    )

    ax = fig.axes[0]

    colors = ["green", "blue", "red", "purple", "orange", "yellow", "black"]

    for i, descent in enumerate(descents):
        x = [price[0] for price in descent.price_history]
        y = [price[1] for price in descent.price_history]
        z = descent.profit_history

        color = colors[i % len(colors)]
        ax.plot(x, y, z, color=color, marker="o", linewidth="0.5", markersize=0.7, label=labels[i])
        ax.plot(x[-1], y[-1], z[-1], "ro")

    ax.legend(loc="upper left")

    return fig


def plot_descent_one_tier(
    X: list | np.ndarray,
    Y: list | np.ndarray,
    descent: GradientDescent,
    title: str,
    azim: int = 225,
    elev: int = 25,
    cmap: str = "plasma",
) -> plt.Figure:

    x = [price[0] for price in descent.price_history]
    z = descent.profit_history

    fig = line_plot(X, Y, "Price", "Expected Profit / Customer", title=title)

    ax = fig.axes[0]

    ax.plot(x, z, color="green", marker="o", linewidth="0.5", markersize=0.7)

    return fig


# ...existing code...

def plot_descent_three_tiers(
    descent: GradientDescent,
    title: str,
    elev: int = 30,
    azim: int = 45,
    figsize: tuple = (10, 8),
    colormap: str = "viridis",
    marker_size: float = 5,
    line_width: float = 1.5,
    show_start_end: bool = True,
    include_colorbar: bool = True
) -> plt.Figure:
    """
    Create a 3D plot of optimization trajectory with points color-coded by profit.

    Parameters
    ----------
    descent : GradientDescent
        The gradient descent object containing price_history and profit_history.
    title : str
        Title of the plot.
    elev : int, optional
        Elevation angle for 3D view (default: 30).
    azim : int, optional
        Azimuthal angle for 3D view (default: 45).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 8)).
    colormap : str, optional
        Colormap to represent profit values (default: 'viridis').
    marker_size : float, optional
        Size of markers for data points (default: 5).
    line_width : float, optional
        Width of the trajectory line (default: 1.5).
    show_start_end : bool, optional
        If True, highlight start and end points (default: True).
    include_colorbar : bool, optional
        If True, add a colorbar for profit values (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the generated 3D plot.
    """
    prices = descent.price_history
    profits = descent.profit_history
    
    
    x = [price[0] for price in prices]
    y = [price[1] for price in prices]
    z = [price[2] for price in prices]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    points = ax.scatter(x, y, z, c=profits, cmap=colormap, s=marker_size)

    for i in range(len(x) - 1):
        ax.plot(
            [x[i], x[i+1]], 
            [y[i], y[i+1]], 
            [z[i], z[i+1]], 
            color=plt.cm.get_cmap(colormap)(i / len(x)),
            linewidth=line_width
        )
    
    if show_start_end:
        ax.scatter(x[0], y[0], z[0], color='red', s=marker_size*3, marker='o', label='Start')
        ax.scatter(x[-1], y[-1], z[-1], color='green', s=marker_size*3, marker='s', label='End')
        ax.legend()

    if include_colorbar:
        cbar = fig.colorbar(points, ax=ax, shrink=0.7)
        cbar.set_label('Profit')
    
    ax.set_xlabel('Tier 1 Price')
    ax.set_ylabel('Tier 2 Price') 
    ax.set_zlabel('Tier 3 Price')
    
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    return fig


def compare_descents_three_tiers(
    descents: list[GradientDescent],
    labels: list[str],
    optimal_prices: tuple[float, float, float],
    title: str,
    elev: int = 30,
    azim: int = 45,
    figsize: tuple = (10, 8),
    colormap: str = "viridis",
    marker_size: float = 3,
    line_width: float = 1.5,
    show_start_end: bool = True,
    include_colorbar: bool = True
)   -> plt.Figure:
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    colors = ["green", "blue", "purple", "orange", "yellow", "black"]

    for i, descent in enumerate(descents):
        x = [price[0] for price in descent.price_history]
        y = [price[1] for price in descent.price_history]
        z = [price[2] for price in descent.price_history]
        profits = descent.profit_history

        color = colors[i % len(colors)]
        points = ax.scatter(x, y, z, c=profits, cmap=colormap, s=marker_size)

        ax.plot(
            x, 
            y,
            z, 
            color=color,
            linewidth=line_width
        )
        if show_start_end:
            ax.scatter(x[-1], y[-1], z[-1], color=color, s=marker_size*10, label=labels[i], marker='s')
    
    ax.scatter(optimal_prices[0], optimal_prices[1], optimal_prices[2], color='red', s=marker_size*10, marker='s', label='Optimal Prices')

    ax.legend(loc="upper left")
    ax.set_xlabel('Tier 1 Price')
    ax.set_ylabel('Tier 2 Price') 
    ax.set_zlabel('Tier 3 Price')
    
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True)
    return fig
    

def descent_title(costs: list[float], lambda_value: float, profit: float, distribution: str, mu: float, sigma: float) -> str:
    title = f"C: {costs}, λ: {lambda_value}, f(v): {distribution}"
    if distribution == 'gaussian':
        return f"{title}, μ: {mu}, σ: {sigma}, F: {profit}"
    else:
        return f"{title}, α: {mu-sigma}, β: {mu+sigma}, F: {profit}"


def descent_label_lr(lr: float) -> str:
    return f"η: {lr}"


def descent_label_lr_profit(lr: float, profit: float) -> str:
    return f"η: {lr}, F: {profit}"
