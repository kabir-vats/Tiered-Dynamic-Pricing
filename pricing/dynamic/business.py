from typing import List
import pandas as pd
from pandas import DataFrame
from pricing.dynamic.customer import Customer


class Business:
    def __init__(self, costs: List[float], customer: Customer) -> None:
        self.costs = costs
        self.net_profit = 0
        self.prices = costs.copy()
        self.tiers = len(costs)
        self.customer = customer
        self.transaction_history = DataFrame(columns=['prices', 'chosen_tier'])

    def sell(self) -> None:
        tier = self.customer.choose_tier(self.costs, self.prices)
        self.net_profit += self.prices[tier] - self.costs[tier]
        new_transaction = pd.DataFrame([[self.prices, tier]],
                                       columns=self.transaction_history.columns)
        self.transaction_history = pd.concat([new_transaction,
                                              self.transaction_history],
                                             ignore_index=True)

    def sell_n(self, n: int) -> None:
        for _ in range(n):
            self.sell()
