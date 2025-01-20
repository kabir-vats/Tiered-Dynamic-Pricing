import pandas as pd
from pandas import DataFrame


class Business:
    def __init__(self, costs, customer):
        self.costs = costs
        self.net_profit = 0
        self.prices = costs.copy()
        self.tiers = len(costs)
        self.customer = customer
        self.transaction_history = DataFrame(columns=['prices', 'chosen_tier'])

    def sell(self):
        tier = self.customer.choose_tier(self.costs, self.prices)
        self.net_profit += self.prices[tier] - self.costs[tier]
        new_transaction = pd.DataFrame([[self.prices, tier]],
                                       columns=self.transaction_history.columns)
        self.transaction_history = pd.concat([new_transaction,
                                              self.transaction_history],
                                             ignore_index=True)

    def sell_n(self, n):
        for _ in range(n):
            self.sell()
