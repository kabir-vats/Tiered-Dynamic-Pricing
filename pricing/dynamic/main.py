from matplotlib import pyplot as plt
import numpy as np

from pricing.dynamic.business import Business
from pricing.dynamic.controller import BatchGradientDescent
from pricing.dynamic.customer import Customer
from pricing.static.optimize import DualAnnealing, GradientDescentAdam
from pricing.static.system import TieredPricingSystem
from pricing.util.simulate import simulate_profits
from pricing.util.visualize import plot_descent


def main():
    np.set_printoptions(legacy='1.25')
    # test_lr()
    C = [1, 4]
    lambda_value = 2/3
    mu = 2
    sigma = 1
    customer = Customer(mu, sigma, lambda_value)
    business = Business(C, customer)
    controller = BatchGradientDescent(business)

    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    descent = GradientDescentAdam(system)

    dual = DualAnnealing(system)
    profits, samples = simulate_profits(system ,n_samples=100)

    controller.maximize()

    descent.maximize()
    dual.maximize()

    print(controller.profit)
    print(descent.profit)
    print(dual.profit)

    print(controller.prices)
    print(system.profit(controller.prices))

    print(descent.prices)
    print(dual.prices)
    plot_descent(samples[0], samples[1], profits, controller, f"{list(system.costs)}")
    # profit: {descent.profit}")
    plt.show()


if __name__ == "__main__":
    main()
