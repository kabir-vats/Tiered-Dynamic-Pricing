import matplotlib.pyplot as plt
import numpy as np

from pricing.static.optimize import GradientDescent
from matplotlib.animation import FuncAnimation

from collections import Counter

from pricing.static.system import TieredPricingSystem


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
    optimal_profit: float,
    title: str,
    elev: int = 30,
    azim: int = 45,
    figsize: tuple = (10, 8),
    colormap: str = "viridis",
    marker_size: float = 1,
    colors=None,
    line_width: float = 1,
    show_start_end: bool = True,
    include_colorbar: bool = True
)   -> plt.Figure:
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if colors is None:
        colors = ["green", "blue", "purple", "orange", "yellow", "black"]

    for i, descent in enumerate(descents):
        x = [price[0] for price in descent.price_history]
        y = [price[1] for price in descent.price_history]
        z = [price[2] for price in descent.price_history]
        profits = descent.profit_history

        color = colors[i % len(colors)]
        points = ax.scatter(x, y, z, color=color, s=marker_size)

        '''ax.plot(
            x, 
            y,
            z, 
            color=color,
            linewidth=line_width
        )'''
        if show_start_end:
            ax.scatter(x[-1], y[-1], z[-1], color=color, s=marker_size*10, label=labels[i], marker='s')
    
    ax.scatter(optimal_prices[0], optimal_prices[1], optimal_prices[2], color='red', s=marker_size*50, marker='s', label=f'Optimal Prices, F: {optimal_profit:.2f}')

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
        return f"{title}, μ: {mu}, σ: {sigma}"
    else:
        return f"{title}, α: {mu-sigma}, β: {mu+sigma}"


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
    colors=None,
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


def compare_profit_history(
    controllers: list,
    controller_labels: list[str],
    system: TieredPricingSystem,
    optimal_profit: float = None,
    figsize: tuple = (12, 7),
    title: str = "Profit Evolution Comparison",
    window_size: int = 1,
    styles: list[str] = None,
    colors: list[str] = None,
    show_final_values: bool = True,
    legend_loc: str = 'best',
    normalize: bool = False,
    y_limit: tuple = None,
    x_limit: tuple = None,
    include_annotations: bool = False
) -> plt.Figure:
    """
    Compare profit histories from multiple controllers during optimization.
    
    Parameters
    ----------
    controllers : list
        List of controller objects (e.g., StochasticGradientDescent) with profit_history attribute.
    controller_labels : list of str
        Labels for each controller in the legend.
    optimal_profit : float, optional
        The optimal/target profit value for reference, if known.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (12, 7)).
    title : str, optional
        Title of the plot (default: "Profit Evolution Comparison").
    window_size : int, optional
        Size of moving average window for smoothing profit curves (default: 1, no smoothing).
    styles : list of str, optional
        Line styles for each controller (default: cycles through basic styles).
    colors : list of str, optional
        Colors for each controller (default: uses consistent color scheme with other functions).
    show_final_values : bool, optional
        If True, show annotations with final profit values (default: True).
    legend_loc : str, optional
        Location of the legend (default: 'best').
    normalize : bool, optional
        If True, normalize profits by dividing by optimal_profit (default: False).
    y_limit : tuple, optional
        Custom y-axis limits as (min, max) (default: None, auto-determined).
    x_limit : tuple, optional
        Custom x-axis limits as (min, max) (default: None, auto-determined).
    include_annotations : bool, optional
        If True, add annotations showing final values and convergence metrics (default: True).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the profit history comparison.
    """
    if len(controllers) != len(controller_labels):
        raise ValueError("Number of controllers must match number of labels")
    
    # Default styles and colors if not provided
    if styles is None:
        styles = ['-', '--', '-.', ':']
    if colors is None:
        colors = ["green", "blue", "purple", "orange", "yellow", "black"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    max_iterations = 0
    final_profits = []
    
    # Plot each controller's profit history
    for i, (controller, label) in enumerate(zip(controllers, controller_labels)):
        if not hasattr(controller, "profit_history") or not controller.profit_history:
            continue
        
        profits = [system.profit(prices) for prices in controller.price_history]
        
        # Normalize if requested and optimal_profit is provided
        if normalize and optimal_profit is not None:
            profits = profits / optimal_profit
        
        # Apply moving average smoothing if window_size > 1
        if window_size > 1:
            # Apply convolution for smoothing
            kernel = np.ones(window_size) / window_size
            smoothed_profits = np.convolve(profits, kernel, mode='valid')
            # Create corresponding x values (iterations)
            iterations = np.arange(window_size-1, len(profits))
            if len(iterations) != len(smoothed_profits):
                iterations = iterations[:len(smoothed_profits)]
        else:
            smoothed_profits = profits
            iterations = np.arange(len(profits))
        
        # Keep track of maximum iterations for plot limits
        max_iterations = max(max_iterations, len(iterations))
        
        # Select style and color with cycling
        style = styles[i % len(styles)]
        color = colors[i % len(colors)]
        
        # Plot the profit history
        ax.plot(iterations, smoothed_profits, 
                linestyle=style, 
                color=color, 
                linewidth=2, 
                label=label)
        
        # Store final profit value for annotations
        final_profits.append(smoothed_profits[-1])
    
    # Add optimal profit line if provided
    if optimal_profit is not None:
        optimal_value = 1.0 if normalize else optimal_profit
        ax.axhline(y=optimal_value, color='red', linestyle='--', alpha=0.7, 
                   label='Optimal Profit' if not normalize else 'Optimal (100%)')
    
    # Add annotations with final values if requested
    if show_final_values and include_annotations:
        for i, (label, final_profit) in enumerate(zip(controller_labels, final_profits)):
            color = colors[i % len(colors)]
            
            # Determine annotation position (avoid overlaps)
            x_pos = max_iterations + max_iterations * 0.01
            
            # Compute percentage of optimal if provided
            if optimal_profit is not None and not normalize:
                percentage = 100 * final_profit / optimal_profit
                ax.annotate(f"{label}: {final_profit:.3f} ({percentage:.1f}%)", 
                           xy=(max_iterations, final_profit),
                           xytext=(x_pos, final_profit),
                           color=color,
                           fontweight='bold',
                           va='center')
            else:
                ax.annotate(f"{label}: {final_profit:.3f}", 
                           xy=(max_iterations, final_profit),
                           xytext=(x_pos, final_profit),
                           color=color,
                           fontweight='bold',
                           va='center')
    
    # Set custom axis limits if provided
    if y_limit is not None:
        ax.set_ylim(y_limit)
    if x_limit is not None:
        ax.set_xlim(x_limit)
    else:
        # Add some padding to the right for annotations
        current_xlim = ax.get_xlim()
        if show_final_values:
            ax.set_xlim(current_xlim[0], current_xlim[1] * 1.2)
    
    # Set labels and title
    ax.set_xlabel('Iteration')
    
    if normalize:
        ax.set_ylabel('Normalized Profit (% of Optimal)')
        # Add percentage ticks on the y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    else:
        ax.set_ylabel('Profit')
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc=legend_loc)
    
    plt.tight_layout()
    return fig


def compare_convergence_metrics(
    controllers: list,
    controller_labels: list[str],
    system: TieredPricingSystem,
    optimal_prices: list[float] = None,
    optimal_profit: float = None,
    figsize: tuple = (14, 8),
    title: str = "Convergence Metrics Comparison",
    colors: list[str] = None,
    metric_types: list[str] = ['price_error', 'profit_ratio', 'iter_to_converge']
):
    """
    Compare various convergence metrics between multiple controllers.
    
    Parameters
    ----------
    controllers : list
        List of controller objects with price_history and profit_history attributes.
    controller_labels : list of str
        Labels for each controller in the chart.
    optimal_prices : list of float, optional
        The optimal prices for reference, used to calculate price error.
    optimal_profit : float, optional
        The optimal profit value for reference, used to calculate profit ratio.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 8)).
    title : str, optional
        Main title of the plot (default: "Convergence Metrics Comparison").
    colors : list of str, optional
        Colors for each controller (default: uses consistent color scheme).
    metric_types : list of str, optional
        Which metrics to include. Options: 'price_error', 'profit_ratio', 
        'iter_to_converge', 'final_profit' (default: includes first three).
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        A Matplotlib Figure object containing the convergence metrics comparison.
    """
    if len(controllers) != len(controller_labels):
        raise ValueError("Number of controllers must match number of labels")
    
    # Default colors if not provided
    if colors is None:
        colors = ["green", "blue", "purple", "orange", "yellow", "black"]
    
    # Determine which metrics to calculate and display
    include_price_error = 'price_error' in metric_types and optimal_prices is not None
    include_profit_ratio = 'profit_ratio' in metric_types and optimal_profit is not None
    include_iter_converge = 'iter_to_converge' in metric_types
    include_final_profit = 'final_profit' in metric_types
    
    # Determine how many metrics we'll display
    num_metrics = sum([include_price_error, include_profit_ratio, 
                       include_iter_converge, include_final_profit])
    
    if num_metrics == 0:
        raise ValueError("No valid metrics to display. Check metric_types and optional parameters.")
    
    # Set up the figure and axes
    fig, axs = plt.subplots(1, num_metrics, figsize=figsize)
    
    # Make axs iterable even if only one subplot
    if num_metrics == 1:
        axs = [axs]
    
    metric_idx = 0
    all_metrics = {}
    
    # Calculate metrics for each controller
    price_errors = []
    profit_ratios = []
    convergence_iters = []
    final_profits = []
    
    for i, controller in enumerate(controllers):
        # Price error metric (if requested and optimal prices available)
        if include_price_error:
            # Calculate normalized Euclidean distance from optimal prices
            if len(controller.price_history[-1]) == len(optimal_prices):
                final_prices = np.array(controller.price_history[-1])
                error = np.linalg.norm(final_prices - np.array(optimal_prices)) / np.linalg.norm(np.array(optimal_prices))
                price_errors.append(error)
            else:
                price_errors.append(np.nan)  # Different dimensionality
        
        # Profit ratio metric (if requested and optimal profit available)
        if include_profit_ratio:
            final_profit = system.profit(controller.price_history[-1])
            ratio = final_profit / optimal_profit
            profit_ratios.append(ratio)
        
        # Iterations to convergence metric (if requested)
        if include_iter_converge:
            # Define convergence as when profit doesn't improve by more than 0.1% for 10 iterations
            profits = np.array([system.profit(prices) for prices in controller.price_history])
            threshold = 0.001  # 0.1% improvement
            window = 10  # Check for 10 consecutive iterations
            
            # Compute relative improvements
            improvements = np.diff(profits) / (profits[:-1] + 1e-10)
            
            # Find where improvements consistently fall below threshold
            converged_at = len(improvements)  # Default to last iteration
            for j in range(len(improvements) - window):
                if np.all(improvements[j:j+window] < threshold):
                    converged_at = j
                    break
            
            convergence_iters.append(converged_at)
        
        # Final profit metric (if requested)
        if include_final_profit:
            final_profits.append(system.profit(controller.price_history[-1]))
    
    # Plot price error metric
    if include_price_error:
        ax = axs[metric_idx]
        bars = ax.bar(controller_labels, price_errors, color=[colors[i % len(colors)] for i in range(len(controllers))])
        ax.set_title('Price Error (lower is better)')
        ax.set_ylabel('Normalized Error')
        ax.set_ylim(0, max(price_errors) * 1.2)
        
        # Add value labels on bars
        for bar, error in zip(bars, price_errors):
            height = bar.get_height()
            ax.annotate(f'{error:.3f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        metric_idx += 1
    
    # Plot profit ratio metric
    if include_profit_ratio:
        ax = axs[metric_idx]
        bars = ax.bar(controller_labels, profit_ratios, color=[colors[i % len(colors)] for i in range(len(controllers))])
        ax.set_title('Profit Ratio (higher is better)')
        ax.set_ylabel('Final Profit / Optimal Profit')
        ax.set_ylim(0, max(profit_ratios) * 1.2)
        
        # Add percentage labels on bars
        for bar, ratio in zip(bars, profit_ratios):
            height = bar.get_height()
            ax.annotate(f'{ratio:.2f} ({ratio*100:.1f}%)',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        metric_idx += 1
    
    # Plot iterations to convergence metric
    if include_iter_converge:
        ax = axs[metric_idx]
        bars = ax.bar(controller_labels, convergence_iters, color=[colors[i % len(colors)] for i in range(len(controllers))])
        ax.set_title('Iterations to Convergence (lower is better)')
        ax.set_ylabel('Iterations')
        ax.set_ylim(0, max(convergence_iters) * 1.2)
        
        # Add value labels on bars
        for bar, iters in zip(bars, convergence_iters):
            height = bar.get_height()
            ax.annotate(f'{iters}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
        
        metric_idx += 1
    
    # Plot final profit metric
    if include_final_profit:
        ax = axs[metric_idx]
        bars = ax.bar(controller_labels, final_profits, color=[colors[i % len(colors)] for i in range(len(controllers))])
        ax.set_title('Final Profit')
        ax.set_ylabel('Profit')
        ax.set_ylim(0, max(final_profits) * 1.2)
        
        # Add value labels on bars
        for bar, profit in zip(bars, final_profits):
            height = bar.get_height()
            ax.annotate(f'{profit:.3f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')
    
    # Set overall title and adjust layout
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    return fig