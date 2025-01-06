from pricing.system import TieredPricingSystem
import numpy as np
from typing import List
import itertools
from tqdm import tqdm


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
