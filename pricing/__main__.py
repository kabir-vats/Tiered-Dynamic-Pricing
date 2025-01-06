from pricing.system import TieredPricingSystem
from pricing.optimize import GradientDescent, DualAnnealing
from pricing.util.simulate import simulate_profits
from pricing.util.visualize import surface_plot
import matplotlib.pyplot as plt


def main():
    C = [1, 4]
    lambda_value = 5/6
    mu = 2
    sigma = 1
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    profits, prices = simulate_profits(system)
    surface_plot(prices[0], prices[1], profits, "Tier 1 Price", "Tier 2 Price",
                 "Profit", "Profit for various prices")
    plt.show()


if __name__ == "__main__":
    main()
