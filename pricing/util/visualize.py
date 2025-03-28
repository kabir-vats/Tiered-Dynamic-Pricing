import matplotlib.pyplot as plt
import numpy as np

from pricing.static.optimize import GradientDescent
from matplotlib.animation import FuncAnimation

from collections import Counter


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


def plot_parameter_history(
    estimator,
    true_params=None,
    figsize=(14, 8),
    title="Parameter Estimation History",
    use_trials_on_x=True,
    include_ci=False,
    confidence_level=0.95,
):
    """
    Visualize the history of parameter estimates from an EfficientGaussianEstimator.
    
    Parameters
    ----------
    estimator : EfficientGaussianEstimator
        The estimator object containing parameter history.
    true_params : tuple, optional
        The true parameter values (mu, sigma, lambda) for reference, if known.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 8)).
    title : str, optional
        Title of the plot (default: "Parameter Estimation History").
    use_trials_on_x : bool, optional
        If True, x-axis shows trial number. If False, uses array index (default: True).
    include_ci : bool, optional
        If True, includes confidence intervals based on particle distribution (default: False).
    confidence_level : float, optional
        Confidence level for intervals if include_ci is True (default: 0.95).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the parameter history plots.
    """
    if not hasattr(estimator, "history") or not estimator.history:
        raise ValueError("Estimator has no history to plot")
    
    history = np.array(estimator.history)
    mu_history = history[:, 0]
    sigma_history = history[:, 1]
    lambda_history = history[:, 2]
    
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Create x-axis values
    x = np.arange(1, len(history) + 1) if use_trials_on_x else np.arange(len(history))
    
    # Plot each parameter
    axs[0].plot(x, mu_history, 'b-', label='μ estimate', linewidth=2)
    axs[1].plot(x, sigma_history, 'g-', label='σ estimate', linewidth=2)
    axs[2].plot(x, lambda_history, 'r-', label='λ estimate', linewidth=2)
    
    # Add confidence intervals if requested
    if include_ci and hasattr(estimator, "particles") and hasattr(estimator, "weights"):
        particles = estimator.particles
        weights = estimator.weights
        
        # Calculate last confidence intervals
        alpha = (1 - confidence_level) / 2
        
        # Sort particles by weight for percentile calculation
        sorted_indices = np.argsort(particles[:, 0])
        sorted_mu = particles[sorted_indices, 0]
        sorted_weights = weights[sorted_indices]
        cumsum_weights = np.cumsum(sorted_weights)
        
        # Find confidence intervals for the last point
        mu_lower = np.interp(alpha, cumsum_weights, sorted_mu)
        mu_upper = np.interp(1 - alpha, cumsum_weights, sorted_mu)
        
        # Same for sigma and lambda
        sorted_indices = np.argsort(particles[:, 1])
        sorted_sigma = particles[sorted_indices, 1]
        sorted_weights = weights[sorted_indices]
        cumsum_weights = np.cumsum(sorted_weights)
        sigma_lower = np.interp(alpha, cumsum_weights, sorted_sigma)
        sigma_upper = np.interp(1 - alpha, cumsum_weights, sorted_sigma)
        
        sorted_indices = np.argsort(particles[:, 2])
        sorted_lambda = particles[sorted_indices, 2]
        sorted_weights = weights[sorted_indices]
        cumsum_weights = np.cumsum(sorted_weights)
        lambda_lower = np.interp(alpha, cumsum_weights, sorted_lambda)
        lambda_upper = np.interp(1 - alpha, cumsum_weights, sorted_lambda)
        
        # Add uncertainty band at the last point
        axs[0].fill_between([x[-1]-0.5, x[-1]+0.5], 
                           [mu_lower, mu_lower], 
                           [mu_upper, mu_upper], 
                           color='blue', alpha=0.2)
        axs[1].fill_between([x[-1]-0.5, x[-1]+0.5], 
                           [sigma_lower, sigma_lower], 
                           [sigma_upper, sigma_upper], 
                           color='green', alpha=0.2)
        axs[2].fill_between([x[-1]-0.5, x[-1]+0.5], 
                           [lambda_lower, lambda_lower], 
                           [lambda_upper, lambda_upper], 
                           color='red', alpha=0.2)
    
    # Add true values if provided
    if true_params is not None:
        true_mu, true_sigma, true_lambda = true_params
        axs[0].axhline(y=true_mu, color='b', linestyle='--', alpha=0.7, label='True μ')
        axs[1].axhline(y=true_sigma, color='g', linestyle='--', alpha=0.7, label='True σ')
        axs[2].axhline(y=true_lambda, color='r', linestyle='--', alpha=0.7, label='True λ')
    
    # Set labels and titles
    axs[0].set_title(title)
    axs[0].set_ylabel('Mean (μ)')
    axs[1].set_ylabel('Std Dev (σ)')
    axs[2].set_ylabel('Lambda (λ)')
    axs[2].set_xlabel('Trial Number' if use_trials_on_x else 'Update Number')
    
    # Add legends
    for ax in axs:
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ...existing code...

def compare_parameter_history(
    estimators: list,
    estimator_labels: list[str],
    true_params=None,
    figsize=(14, 10),
    title="Comparison of Parameter Estimation Histories",
    use_trials_on_x=True,
    confidence_intervals=False,
    confidence_level=0.95,
    styles=None,
    colors=None,
    legend_loc='best'
):
    """
    Compare parameter estimation histories from multiple estimators.
    
    Parameters
    ----------
    estimators : list of EfficientGaussianEstimator
        List of estimator objects containing parameter histories to compare.
    estimator_labels : list of str
        Labels for each estimator in the legend.
    true_params : tuple, optional
        The true parameter values (mu, sigma, lambda) for reference, if known.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 10)).
    title : str, optional
        Main title of the plot (default: "Comparison of Parameter Estimation Histories").
    use_trials_on_x : bool, optional
        If True, x-axis shows trial number. If False, uses array index (default: True).
    confidence_intervals : bool, optional
        If True, includes confidence intervals for the last estimator (default: False).
    confidence_level : float, optional
        Confidence level for intervals if include_ci is True (default: 0.95).
    styles : list of str, optional
        Line styles for each estimator (default: cycles through basic styles).
    colors : list of str, optional
        Colors for each estimator (default: cycles through matplotlib default colors).
    legend_loc : str, optional
        Location of the legend (default: 'best').
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the comparison plots.
    """
    if len(estimators) != len(estimator_labels):
        raise ValueError("Number of estimators must match number of labels")
    
    # Default styles and colors if not provided
    if styles is None:
        styles = ['-', '--', '-.', ':']
    if colors is None:
        colors = ["green", "blue", "purple", "orange", "yellow", "black"]
    
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Parameter names for labels
    param_names = ['Mean (μ)', 'Std Dev (σ)', 'Lambda (λ)']
    
    # Plot each estimator's history
    for i, (estimator, label) in enumerate(zip(estimators, estimator_labels)):
        if not hasattr(estimator, "history") or not estimator.history:
            continue
            
        history = np.array(estimator.history)
        
        # Make sure history has enough data
        if len(history) < 2:
            continue
            
        # Create x-axis values for this estimator
        x = np.arange(1, len(history) + 1) if use_trials_on_x else np.arange(len(history))
        
        # Select style and color with cycling
        style = styles[i % len(styles)]
        color = colors[i % len(colors)]
        
        # Plot each parameter
        for j in range(3):  # 0: mu, 1: sigma, 2: lambda
            axs[j].plot(x, history[:, j], 
                       linestyle=style, 
                       color=color, 
                       linewidth=2, 
                       label=label if j == 0 else None)  # Only add label in first subplot
        
        # Add confidence intervals for the last estimator if requested
        if confidence_intervals and i == len(estimators) - 1 and hasattr(estimator, "particles") and hasattr(estimator, "weights"):
            particles = estimator.particles
            weights = estimator.weights
            
            # Calculate confidence intervals
            alpha = (1 - confidence_level) / 2
            
            for j, param_idx in enumerate([0, 1, 2]):  # mu, sigma, lambda
                sorted_indices = np.argsort(particles[:, param_idx])
                sorted_param = particles[sorted_indices, param_idx]
                sorted_weights = weights[sorted_indices]
                cumsum_weights = np.cumsum(sorted_weights)
                
                param_lower = np.interp(alpha, cumsum_weights, sorted_param)
                param_upper = np.interp(1 - alpha, cumsum_weights, sorted_param)
                
                # Add uncertainty band at the last point
                axs[j].fill_between([x[-1]-0.5, x[-1]+0.5], 
                                   [param_lower, param_lower], 
                                   [param_upper, param_upper], 
                                   color=color, alpha=0.2)
    
    # Add true parameters if provided
    if true_params is not None:
        true_mu, true_sigma, true_lambda = true_params
        true_values = [true_mu, true_sigma, true_lambda]
        
        for j, val in enumerate(true_values):
            axs[j].axhline(y=val, color='k', linestyle='--', alpha=0.7, label='True Value' if j == 0 else None)
    
    # Set labels and titles
    axs[0].set_title(title)
    for j, name in enumerate(param_names):
        axs[j].set_ylabel(name)
        axs[j].grid(True, alpha=0.3)
    
    axs[2].set_xlabel('Trial Number' if use_trials_on_x else 'Update Number')
    
    # Add legend to first subplot only
    axs[0].legend(loc=legend_loc)
    
    plt.tight_layout()
    return fig

def plot_choice_distribution(
    estimator, 
    figsize=(12, 6),
    title="Customer Choice Distribution"
):
    """
    Visualize the distribution of customer choices from the estimator history.
    
    Parameters
    ----------
    estimator : EfficientGaussianEstimator
        The estimator object containing trial history.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 6)).
    title : str, optional
        Title of the plot (default: "Customer Choice Distribution").
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the choice distribution visualization.
    """
    if not hasattr(estimator, "prev_trials") or not estimator.prev_trials:
        raise ValueError("Estimator has no trial history to plot")
    
    # Count choices across all trials
    all_choices = []
    for trial in estimator.prev_trials:
        all_choices.extend(trial.choices)
    
    choice_counts = Counter(all_choices)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get tier names and counts
    tiers = sorted(choice_counts.keys())
    counts = [choice_counts[t] for t in tiers]
    
    # Generate tier labels (including "No Purchase" for tier 0)
    tier_labels = ["No Purchase" if t == 0 else f"Tier {t}" for t in tiers]
    
    # Create bar chart
    bars = ax.bar(tier_labels, counts, color='skyblue')
    
    # Add count labels above each bar
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel('Customer Choice')
    ax.set_ylabel('Number of Customers')
    
    # Add percentage on a secondary y-axis
    ax2 = ax.twinx()
    percentages = [100 * count / sum(counts) for count in counts]
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Percentage (%)')
    
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax2.annotate(f'{percentage:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 20),  # 20 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    return fig