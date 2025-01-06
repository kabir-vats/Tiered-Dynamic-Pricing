from pricing.system import TieredPricingSystem
from pricing.optimize import GradientDescent, DualAnnealing # noqa
from pricing.util.simulate import simulate_optimal_profits
from pricing.util.visualize import surface_plot
import matplotlib.pyplot as plt


def main():
    C = [1, 4]
    lambda_value = 5/6
    mu = 2
    sigma = 1
    bounds = [[0.1, 10], [0.1, 10]]
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    profits, prices, costs = simulate_optimal_profits(system, bounds, 10)
    surface_plot(costs[0], costs[1], profits, "Tier 1 Cost", "Tier 2 Cost",
                 "Profit", "Optimal profit for various costs") 
    plt.show()


if __name__ == "__main__":
    main()
