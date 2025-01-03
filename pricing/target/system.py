import numpy as np
import sys
from scipy.stats import norm


class tiered_pricing_system:
    """
    Tiered Pricing System

    Parameters
    ----------
    costs : list
        List of costs for each tier
    tiers : int
        Number of tiers
    scaling_param : float
        Scaling parameter for the tiered pricing system
    mu : float
        Mean of the Gaussian distribution or the center of the uniform distribution
    sigma : float
        Standard deviation of the Gaussian distribution or half width of the
        uniform distribution
    pdf_type : str
        Type of probability density function (pdf) to use. 'uniform' or 'gaussian'
    """
    def __init__(self, costs, tiers, scaling_param, mu, sigma, pdf_type='uniform'):
        self.costs = costs
        self.tiers = tiers
        self.scaling_param = scaling_param
        self.mu = mu
        self.sigma = sigma
        self.pdf_type = pdf_type

    def calculate_intervals(self, prices_unsorted):
        """
        Calculate the intervals of valuation parameter for which each tier
        is the optimal choice

        Parameters
        ----------
        prices_unsorted : list
            List of prices for each tier

        Returns
        -------
        intervals_ordered : list
            List of tuples containing the intervals of valuation parameter
            for which each tier is the optimal choice
        """

        # We need to argsort by cost first because function logic relies on sequence of
        # increasing slope lines (slope is cost)
        sorted_indices = np.argsort(self.costs)
        costs = np.array(self.costs)[sorted_indices]
        prices = np.array(prices_unsorted)[sorted_indices]

        thresholds = []
        thresholds.append(-sys.float_info.max)

        for i in range(self.tiers):
            if i == 0:
                thresholds.append(prices[0]/costs[0])

            else:
                intersection = (prices[i] - prices[i-1]) / (
                    costs[0] * (
                        (costs[i] / costs[0]) ** (self.scaling_param) -
                        (costs[i-1] / costs[0]) ** (self.scaling_param)
                    )
                )

                j = 0

                while intersection < thresholds[i - j]:
                    j += 1
                    if i == j:
                        intersection = prices[i] / (
                            costs[0] * (costs[i] / costs[0]) ** (self.scaling_param)
                        )
                        break
                    else:
                        intersection = (prices[i] - prices[i-j-1]) / (
                            costs[0] * (
                                (costs[i]/costs[0])**(self.scaling_param) -
                                (costs[i-j-1]/costs[0])**(self.scaling_param)
                            )
                        )
                thresholds.append(intersection)

        for i in range(self.tiers):
            if (thresholds[self.tiers - i] < thresholds[self.tiers-i-1]):
                thresholds[self.tiers-i-1] = thresholds[self.tiers - i]

        intervals = []
        for i in range(self.tiers):
            intervals.append((thresholds[i], thresholds[i+1]))

        intervals.append((thresholds[len(thresholds)-1], sys.float_info.max))

        intervals_ordered = [None] * len(intervals)
        intervals_ordered[0] = intervals[0]
        for i, idx in enumerate(sorted_indices):
            intervals_ordered[idx+1] = intervals[i+1]

        return intervals_ordered

    def tier_probabilities(self, prices):
        """
        Calculate the probability that each tier is the optimal choice

        Parameters
        ----------
        prices : list
            List of prices for each tier

        Returns
        -------
        probabilities : list
            List of probabilities that each tier is the optimal choice
        """
        probabilities = []
        intervals = self.calculate_intervals(prices)

        if (self.pdf_type == 'uniform'):
            start = self.mu - self.sigma
            end = self.mu + self.sigma
            point_prob = 1/(end - start)
            for i in range(self.tiers+1):
                probabilities.append(max((min(intervals[i][1], end)
                                          - max(intervals[i][0], start))*point_prob, 0))
        elif (self.pdf_type == 'gaussian'):
            for i in range(self.tiers+1):
                probabilities.append(norm.cdf(intervals[i][1], self.mu, self.sigma)
                                     - norm.cdf(intervals[i][0], self.mu, self.sigma))
        else:
            raise ValueError('pdf_type must be uniform or gaussian')

        return probabilities

    def profit(self, prices):
        """
        Calculate the profit of the tiered pricing system

        Parameters
        ----------
        prices : list
            List of prices for each tier

        Returns
        -------
        profit : float
            Profit of the tiered pricing system for the given prices
        """
        probabilities = self.tier_probabilities(prices)

        profits = [pr * (p - c) for (p, c, pr) in zip(prices, self.costs,
                                                      probabilities[1:])]
        return sum(profits)
