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
    C = [1, 4]
    lambda_value = 5 / 6
    mu = 2
    sigma = 1
    customer = Customer(mu, sigma, lambda_value)
    business = Business(C, customer)
    controller = BatchGradientDescent(business)

    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    descent = GradientDescentAdam(system)

    dual = DualAnnealing(system)

    controller.maximize()

    descent.maximize()
    dual.maximize()
    profits, samples = simulate_profits(system, n_samples=100)

    print(controller.mock_system.mu)
    print(controller.mock_system.sigma)
    print(controller.mock_system.lam)
    print(controller.estimator.a_mean)
    print(controller.estimator.b_mean)
    print(controller.estimator.lambda_mean)
    print(controller.profit)
    print(descent.profit)
    print(dual.profit)

    print(controller.prices)
    print(system.profit(controller.prices))

    print(descent.prices)
    print(dual.prices)
    plot_descent_two_tiers(
        samples[0],
        samples[1],
        profits,
        controller,
        f"Costs: {list(system.costs)} Lambda: {lambda_value} Prof: {controller.profit}",
    )
    # profit: {descent.profit}")
    plt.show()


if __name__ == "__main__":
    main()
