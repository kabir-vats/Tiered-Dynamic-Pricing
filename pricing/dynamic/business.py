from typing import List
import pandas as pd
from pandas import DataFrame
from pricing.dynamic.customer import Customer


class Business:
    def __init__(self, costs: List[float], customer: Customer) -> None:
        self.costs = costs
        self.net_profit = 0
        self.tiers = len(costs)
        self.customer = customer
        self.transaction_history = DataFrame(columns=['prices', 'chosen_tier'])

    def sell(self, prices: List[float]) -> pd.DataFrame:
        tier = self.customer.choose_tier(self.costs, prices)
        if tier > 0:
            self.net_profit += prices[tier-1] - self.costs[tier-1]
        '''new_transaction = pd.DataFrame([[prices, tier]],
                                       columns=self.transaction_history.columns)
        self.transaction_history = pd.concat([new_transaction,
                                              self.transaction_history],
                                             ignore_index=True)
        return new_transaction'''

    def sell_n(self, prices: List[float], n: int) -> pd.DataFrame:
        profit_before = self.net_profit
        for _ in range(n):
            self.sell(prices)
        return (self.net_profit - profit_before)/n, 1 #, self.transaction_history.tail(n)
