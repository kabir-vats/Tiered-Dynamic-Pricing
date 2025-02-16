from typing import List
from pandas import DataFrame
from pricing.dynamic.customer import Customer


class Business:
    """
    A business with given costs and a customer.

    Parameters
    ----------
    costs : List[float]
        Cost thresholds for the tiers.
    customer : Customer
        Customer object with distribution parameters.
    """

    def __init__(self, costs: List[float], customer: Customer) -> None:
        self.costs = costs
        self.net_profit = 0
        self.tiers = len(costs)
        self.customer = customer
        self.transaction_history = DataFrame(columns=["prices", "chosen_tier"])

    def sell(self, prices: List[float]) -> tuple[float, int]:
        """
        Process the sale for a single customer at provided prices.

        Parameters
        ----------
        prices : List[float]
            Price list for each available tier.

        Updates
        -------
        self.net_profit : float
            Increased based on the chosen tier's profit margin.
        """
        tier = self.customer.choose_tier(self.costs, prices)
        if tier > 0:
            self.net_profit += prices[tier - 1] - self.costs[tier - 1]

        return prices[tier - 1] - self.costs[tier - 1], tier
        """new_transaction = pd.DataFrame([[prices, tier]],
                                       columns=self.transaction_history.columns)
        self.transaction_history = pd.concat([new_transaction,
                                              self.transaction_history],
                                             ignore_index=True)
        return new_transaction"""

    def sell_n(self, prices: List[float], n: int) -> tuple:
        """
        Process multiple sales, returning the average profit gained.

        Parameters
        ----------
        prices : List[float]
            Price list for each tier.
        n : int
            Number of customers to sell to.

        Returns
        -------
        tuple
            A (float, int) tuple where the first element is the average profit
            per customer, and the second element is a dummy placeholder value.

        Updates
        -------
        self.net_profit : float
            Incremented over multiple sales.
        """
        profit_before = self.net_profit
        for _ in range(n):
            self.sell(prices)
        return (
            self.net_profit - profit_before
        ) / n, 1  # , self.transaction_history.tail(n)
