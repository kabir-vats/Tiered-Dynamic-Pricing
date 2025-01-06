from pricing.system import TieredPricingSystem
from pricing.optimize import GradientDescent, DualAnnealing # noqa
from pricing.util.simulate import simulate_optimal_profits
from pricing.util.visualize import surface_plot
from pricing.experiment.optimize_descent import test_descent
import matplotlib.pyplot as plt


def main():
    C = [1, 4]
    lambda_value = 3/6
    mu = 2
    sigma = 1
    bounds = [[0.1, 10], [0.1, 10]]
    num_samples = 10
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    diff, _, _, _ = test_descent(system, bounds, num_samples)
    plt.plot(diff)
    plt.show()


if __name__ == "__main__":
    main()
