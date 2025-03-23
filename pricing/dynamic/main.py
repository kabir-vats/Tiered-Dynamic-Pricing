from matplotlib import pyplot as plt
import numpy as np

from pricing.dynamic.business import Business
from pricing.dynamic.controller import StochasticGradientDescent
from pricing.dynamic.customer import Customer
from pricing.static.optimize import DualAnnealing, GradientDescent
from pricing.static.system import TieredPricingSystem
from pricing.util.simulate import simulate_profits
from pricing.util.visualize import plot_descent_two_tiers


def main():
    np.set_printoptions(legacy="1.25")
    # test_lr()
    C = [1, 3]
    lambda_value = 1/3
    mu = 4
    sigma = 3
    customer = Customer(mu, sigma, lambda_value, pdf_type="gaussian")
    business = Business(C, customer)
    controller = StochasticGradientDescent(business, max_iters=300, lr=0.02, batch_size=1, pdf_type="gaussian")

    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma, pdf_type="gaussian")
    descent = GradientDescent(system)

    dual = DualAnnealing(system)

    controller.maximize()

    descent.maximize()
    dual.maximize()
    profits, samples = simulate_profits(system, n_samples=100)

    print(f"Profit controller believes it achieved: {controller.profit}")
    print(f"Profit controller actually achieved {system.profit(controller.prices)}")
    print(f"Gradient descent on ideal system profit: {descent.profit}")
    print(f"Dual Annealing Profit {dual.profit}")

    print(f"Gradient Descent Prices {descent.prices}")
    print(f"Dual Annealing Prices {dual.prices}")
    print(f"Controller Prices {controller.prices}")

    print(
        f"Ideal proportion of customers choosing each tier {system.tier_probabilities(dual.prices)}"
    )

    print(
        f"Actual proportion of customers choosing each tier {system.tier_probabilities(controller.prices)}"
    )

    print(
        f"Proportion controller thought were choosing each tier {controller.mock_system.tier_probabilities(controller.prices)}"
    )

    # print(controller.estimator.particles)
    # print(controller.estimator.weights)
    print(max(controller.estimator.weights))
    print(controller.estimator.particles[np.argmax(controller.estimator.weights)])
    print(len(controller.estimator.weights[controller.estimator.weights==0]))
    print(controller.estimator.particles[:10])
    print(
        controller.estimator.mu_mean,
        controller.estimator.sigma_mean,
        controller.estimator.lambda_mean,
    )
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
