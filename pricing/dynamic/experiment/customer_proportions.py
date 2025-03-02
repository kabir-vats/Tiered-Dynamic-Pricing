from collections import Counter
from pricing.dynamic.customer import Customer
from pricing.dynamic.business import Business
from pricing.static.system import TieredPricingSystem


def test_decisions():
    C = [2, 6]
    lambda_value = 5 / 6
    mu = 3
    sigma = 3
    customer = Customer(mu, sigma, lambda_value)
    business = Business(C, customer)
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    prices = [3, 5]

    tier_choices = []

    for _ in range(100):
        profit, choice = business.sell(prices)
        tier_choices.append(choice)

    print(Counter(tier_choices))
    print(system.tier_probabilities(prices))


if __name__ == "__main__":
    test_decisions()
