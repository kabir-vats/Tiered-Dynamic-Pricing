from typing import List
import numpy as np


class Customer:
    """
    Customer with given distribution of valuations

    Parameters
    ----------
    mu : float
        Mean value for the distribution.
    sigma : float
        Standard deviation for the distribution.
    lam : float
        Lambda parameter for utility exponent.
    pdf_type : str
        Type of distribution ('uniform' or 'normal').
    """

    def __init__(self, mu: float, sigma: float, lam: float,
                 pdf_type: str = 'uniform') -> None:
        self.__mu = mu
        self.__sigma = sigma
        self.__lam = lam
        self.pdf_type = pdf_type

    def utility(self, cost: float, base_cost: float, price: float,
                valuation_param: float) -> float:
        """
        Calculate the utility of a customer given a cost and a price.

        Parameters
        ----------
        cost : float
            The cost of the customer.
        base_cost : float
            The base cost of the tier.
        price : float
            The price of the tier.
        valuation_param : float
            Customer's valuation parameter.

        Returns
        -------
        utility : float
            The utility of the customer.
        """

        return valuation_param * (cost / base_cost)**self.__lam - price

    def choose_tier(self, costs: List[float], prices: List[float]) -> int:
        """
        Choose the tier that the customer belongs to based on their cost and the prices.

        Parameters
        ----------
        costs : List[float]
            The costs for each tier.
        prices : List[float]
            The prices for each tier.

        Returns
        -------
        tier : int
            The tier that the customer belongs to.
        """

        if self.pdf_type == 'uniform':
            valuation_param = np.random.uniform(self.__mu - self.__sigma,
                                                self.__mu + self.__sigma)
        else:
            valuation_param = np.random.normal(self.__mu, self.__sigma)

        base_cost = min(costs)

        utilities = [0] + [self.utility(cost, base_cost, price, valuation_param)
                           for cost, price in zip(costs, prices)]
        return np.argmax(utilities)
