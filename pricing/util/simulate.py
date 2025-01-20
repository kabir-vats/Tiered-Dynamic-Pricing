from pricing.static.system import TieredPricingSystem
import numpy as np
from typing import List
import itertools
from tqdm import tqdm
from pricing.static.optimize import DualAnnealing


def simulate_profits(system: TieredPricingSystem,
                     bounds: List = None, n_samples: int = 50):
    """
    Calculate profits for prices at sample points across an interval

    Parameters
    ----------
    system : TieredPricingSystem
        The system to sample prices on

    bounds : List[Tuple], optional
        The bounds for each tier's price, expressed as a list of
        tuples of lower, higher for the price of  the tier at each index

    n_samples : int, optional
        The number of samples per price.

    Returns
    -------
    profits : List[float]
        Profits evaluated at each combination of prices
    samples : List[List[float]]
        List of the price each tier was evaluated at
    """
    if bounds is None:
        bounds = [(cost * (system.mu + system.sigma),
                   cost * (system.mu - system.sigma)) for cost in system.costs]
    elif len(bounds) != len(system.costs):
        raise Exception("Illegal bounds for system")

    samples = [np.linspace(bound[0], bound[1], num=n_samples) for bound in bounds]

    sample_prices = list(itertools.product(*samples))
    profits = []

    for prices in tqdm(sample_prices):
        profits.append(system.profit(prices))

    return profits, samples


def simulate_optimal_profits(system, bounds: List, n_samples: int,
                             optimizer=DualAnnealing):
    """
    Calculate optimal profits and prices for costs at sample points across an interval

    Parameters
    ----------
    system : TieredPricingSystem
        The system to sample costs on

    bounds : List[Tuple]
        The bounds for each tier's price, expressed as a list of
        tuples of lower, higher for the price of  the tier at each index

    n_samples : int
        The number of samples per price.

    optimizer : DualAnnealing | GradientDescent, opt
        The optimization method, default is DualAnnealing

    Returns
    -------
    profits : List[float]
        Optimal profits evaluated at each combination of costs
    prices : List[list[float]]
        Optimal prices evaluated at each combination of costs
    costs : List[List[float]]
        List of the cost each tier was evaluated at
    """
    samples = [np.linspace(bound[0], bound[1], num=n_samples) for bound in bounds]

    sample_costs = list(itertools.product(*samples))
    prices = []
    profits = []

    for costs in tqdm(sample_costs):
        system.costs = costs
        opt = optimizer(system)
        opt.maximize()
        prices.append(opt.prices.copy())
        profits.append(opt.profit)
    return profits, prices, samples
