import numpy as np
import sys
from scipy.stats import norm
from typing import List, Tuple


class TieredPricingSystem:
    """
    A system for tiered pricing, supporting uniform and Gaussian distributions
    for determining tier probabilities and profit calculation.

    Parameters
    ----------
    costs : List[float]
        List of costs for each tier.
    tiers : int
        Number of tiers in the pricing system.
    lam : float
        Scaling parameter that influences pricing behavior.
    mu : float
        Mean of the distribution (center of uniform or mean of Gaussian).
    sigma : float
        Spread of the distribution (half-width for uniform,
        standard deviation for Gaussian).
    pdf_type : str, optional
        Type of probability density function to use ('uniform' or 'gaussian').
        Default is 'uniform'.
    """

    def __init__(
        self,
        costs: List[float],
        tiers: int,
        lam: float,
        mu: float,
        sigma: float,
        pdf_type: str = "uniform",
    ) -> None:
        self.costs = costs
        self.utils = [min(costs) * (cost / min(costs)) ** lam for cost in costs]
        self.tiers = tiers
        self.lam = lam
        self.mu = mu
        self.sigma = sigma
        self.pdf_type = pdf_type

    def update_parameters(self, mu: float, sigma: float, lam: float) -> None:
        """
        Update the parameters of the system.

        Parameters
        ----------
        mu : float
            Mean of the distribution (center of uniform or mean of Gaussian).
        sigma : float
            Spread of the distribution (half-width for uniform,
            standard deviation for Gaussian).
        lam : float
            Scaling parameter that influences pricing behavior.
        """
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.utils = [
            min(self.costs) * (cost / min(self.costs)) ** lam for cost in self.costs
        ]

    def calculate_intervals(
        self, prices_unsorted: List[float]
    ) -> List[Tuple[float, float]]:
        """
        Calculate intervals of the valuation parameter for which each tier is optimal.

        Parameters
        ----------
        prices_unsorted : List[float]
            List of unsorted prices for each tier.

        Returns
        -------
        intervals_ordered : List[Tuple[float, float]]
            List of intervals (tuples) representing the range of valuation parameters
            where each tier is optimal.
        """
        sorted_indices = np.argsort(self.utils)
        utils = np.array(self.utils)[sorted_indices]
        prices = np.array(prices_unsorted)[sorted_indices]

        thresholds = [-sys.float_info.max]

        for i in range(self.tiers):
            if i == 0:
                thresholds.append(prices[0] / utils[0])
            else:
                intersection = (prices[i] - prices[i - 1]) / (utils[i] - utils[i - 1])
                j = 0
                while intersection < thresholds[i - j]:
                    j += 1
                    if i == j:
                        intersection = prices[i] / utils[i]
                        break
                    else:
                        intersection = (prices[i] - prices[i - j - 1]) / (
                            utils[i] - utils[i - j - 1]
                        )
                thresholds.append(intersection)

        for i in range(self.tiers):
            if thresholds[self.tiers - i] < thresholds[self.tiers - i - 1]:
                thresholds[self.tiers - i - 1] = thresholds[self.tiers - i]

        intervals = [(thresholds[i], thresholds[i + 1]) for i in range(self.tiers)]
        intervals.append((thresholds[-1], sys.float_info.max))

        intervals_ordered = [None] * len(intervals)
        intervals_ordered[0] = intervals[0]
        for i, idx in enumerate(sorted_indices):
            intervals_ordered[idx + 1] = intervals[i + 1]

        return intervals_ordered

    def tier_probabilities(self, prices: List[float]) -> List[float]:
        """
        Calculate probabilities for each tier being the optimal choice.

        Parameters
        ----------
        prices : List[float]
            List of prices for each tier.

        Returns
        -------
        probabilities : List[float]
            List of probabilities corresponding to each tier.
        """
        probabilities = []
        intervals = self.calculate_intervals(prices)

        if self.pdf_type == "uniform":
            start, end = self.mu - self.sigma, self.mu + self.sigma
            point_prob = 1 / (end - start)
            for interval in intervals:
                prob = max(
                    (min(interval[1], end) - max(interval[0], start)) * point_prob, 0
                )
                probabilities.append(prob)
        elif self.pdf_type == "gaussian":
            for interval in intervals:
                prob = norm.cdf(interval[1], self.mu, self.sigma) - norm.cdf(
                    interval[0], self.mu, self.sigma
                )
                probabilities.append(prob)
        else:
            raise ValueError("pdf_type must be 'uniform' or 'gaussian'.")

        return probabilities

    def profit(self, prices: List[float]) -> float:
        """
        Calculate the profit of the tiered pricing system for given prices.

        Parameters
        ----------
        prices : List[float]
            List of prices for each tier.

        Returns
        -------
        profit : float
            Total profit calculated from the given prices.
        """
        probabilities = self.tier_probabilities(prices)
        profits = [
            (pr * (p - c)) for p, c, pr in zip(prices, self.costs, probabilities[1:])
        ]
        return sum(profits)
