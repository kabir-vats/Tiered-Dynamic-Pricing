import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from pricing.static.optimize import GradientDescent, DualAnnealing, GradientDescentAdam
from pricing.static.system import TieredPricingSystem
from pricing.util.simulate import simulate_profits
from pricing.util.visualize import plot_descent_two_tiers, plot_descent_one_tier, compare_descents_two_tiers


def test_descent(system, bounds, n_samples):
    samples = [np.linspace(bound[0], bound[1], num=n_samples) for bound in bounds]
    sample_costs = list(itertools.product(*samples))
    profits_descent = []
    profits_dual = []

    for costs in tqdm(sample_costs):
        system.costs = costs
        descent = GradientDescentAdam(system)
        dual = DualAnnealing(system)

        descent.maximize()
        profits_descent.append(descent.profit)

        dual.maximize()
        profits_dual.append(dual.profit)

    profits_diff = [x-y for x, y in zip(profits_dual, profits_descent)]

    return profits_diff, profits_dual, profits_descent, sample_costs


def test_lr():
    C = [1, 4] # dummy values for costs
    lambda_value = 5/6
    mu = 2
    sigma = 1
    bounds = [[0.1, 1000], [0.1, 1000]]
    num_samples = 10
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    diff, _, _, sample_costs = test_descent(system, bounds, num_samples)
    plt.plot(diff)
    plt.title("Maximum profit miss, learning rate min cost div 3")
    plt.show()
    plt.pause(0.001)
    
    for i in range(len(diff)):
        difference = diff[i]    
        if difference > 0.01:
            system.costs = sample_costs[i]
            profits, samples = simulate_profits(system)
            descent = GradientDescentAdam(system)
            descent.maximize()
            plot_descent_two_tiers(samples[0], samples[1], profits, descent, f"{list(sample_costs[i])} lambda: {lambda_value} profit: {descent.profit} difference: {difference}")
            plt.show()
            plt.pause(0.001)
    plt.show()


def main():
    np.set_printoptions(legacy='1.25')
    # test_lr()
    C = [1, 4]
    lambda_value = 2/3
    mu = 2
    sigma = 1
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    profits, samples = simulate_profits(system ,n_samples=100)

    descent1 = GradientDescentAdam(system, gradient_method='numerical', max_iters=100)
    descent1.maximize()

    descent2 = GradientDescentAdam(system, gradient_method='analytic', max_iters=100)
    descent2.maximize()

    dual = DualAnnealing(system)
    dual.maximize()

    print(descent1.profit)
    print(dual.profit)

    print(system.tier_probabilities(descent1.prices))
    print(system.tier_probabilities(dual.prices))

    print(descent1.prices)
    print(dual.prices)
    compare_descents_two_tiers(samples[0], samples[1], profits, descent1, descent2, f"costs: {list(system.costs)} lambda: {lambda_value} profit: {descent1.profit}")
    plt.show()


if __name__ == "__main__":
    main()
