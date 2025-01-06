from pricing.system import TieredPricingSystem
from pricing.optimize import GradientDescent, DualAnnealing


def main():
    C = [1,4]
    lambda_value = 5/6
    mu = 2
    sigma = 1
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    descent = GradientDescent(system)
    descent.maximize(system)
    print(descent.profit)


if __name__ == "__main__":
    main()
