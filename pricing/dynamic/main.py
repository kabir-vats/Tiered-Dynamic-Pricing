from matplotlib import pyplot as plt
import numpy as np

from pricing.dynamic.business import Business
from pricing.dynamic.controller import BatchGradientDescent
from pricing.dynamic.customer import Customer
from pricing.static.optimize import DualAnnealing, GradientDescentAdam
from pricing.static.system import TieredPricingSystem
from pricing.util.simulate import simulate_profits
from pricing.util.visualize import plot_descent_two_tiers


def main():
    np.set_printoptions(legacy="1.25")
    # test_lr()
    C = [2,4,6,10]
    lambda_value = 5 / 6
    mu = 3
    sigma = 3
    customer = Customer(mu, sigma, lambda_value)
    business = Business(C, customer)
    controller = BatchGradientDescent(business, batch_size=15)

    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    descent = GradientDescentAdam(system)

    dual = DualAnnealing(system)

    controller.maximize()

    descent.maximize()
    dual.maximize()
    # profits, samples = simulate_profits(system, n_samples=100)

    print(f'Profit controller believes it achieved: {controller.profit}')
    print(f'Profit controller actually achieved {system.profit(controller.prices)}')
    print(f'Gradient descent on ideal system profit: {descent.profit}')
    print(f'Dual Annealing Profit {dual.profit}')

    print(f'Gradient Descent Prices {descent.prices}')
    print(f'Dual Annealing Prices {dual.prices}')
    print(f'Controller Prices {controller.prices}')

    print(f'Ideal proportion of customers choosing each tier {system.tier_probabilities(dual.prices)}')

    print(f'Actual proportion of customers choosing each tier {system.tier_probabilities(controller.prices)}')

    '''plot_descent_two_tiers(
        samples[0],
        samples[1],
        profits,
        controller,
        f"Costs: {list(system.costs)} Lambda: {lambda_value} Prof: {controller.profit}",
    )
    # profit: {descent.profit}")
    plt.show()'''


if __name__ == "__main__":
    main()
