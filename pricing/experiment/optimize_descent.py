import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from pricing.optimize import GradientDescent, DualAnnealing
from pricing.system import TieredPricingSystem
from pricing.util.simulate import simulate_profits
from pricing.util.visualize import plot_descent


def test_descent(system, bounds, n_samples):
    samples = [np.linspace(bound[0], bound[1], num=n_samples) for bound in bounds]
    sample_costs = list(itertools.product(*samples))
    profits_descent = []
    profits_dual = []

    for costs in tqdm(sample_costs):
        system.costs = costs
        descent = GradientDescent(system)
        dual = DualAnnealing(system)

        descent.maximize()
        profits_descent.append(descent.profit)

        dual.maximize()
        profits_dual.append(dual.profit)

    profits_diff = [x-y for x, y in zip(profits_dual, profits_descent)]

    return profits_diff, profits_dual, profits_descent, sample_costs


def main():
    plt.ion()
    np.set_printoptions(legacy='1.25')
    C = [1, 4]
    lambda_value = 3/6
    mu = 2
    sigma = 1
    bounds = [[0.1, 100], [0.1, 100]]
    num_samples = 30
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma)
    diff, _, _, sample_costs = test_descent(system, bounds, num_samples)
    plt.plot(diff)
    plt.title("Maximum profit miss, learning rate min cost div 3")
    plt.show()
    plt.pause(0.001)
    '''
    for i in range(len(diff)):
        difference = diff[i]
        if difference > 0.1:
            system.costs = sample_costs[i]
            profits, samples = simulate_profits(system)
            descent = GradientDescent(system)
            descent.maximize()
            plot_descent(samples[0], samples[1], profits, descent, f"{list(sample_costs[i])} profit: {descent.profit} learning rate: {min(system.costs)/2}")
            plt.show()
            plt.pause(0.001)
            descent = GradientDescent(system, lr=min(system.costs)/3)
            descent.maximize()
            plot_descent(samples[0], samples[1], profits, descent, f"{list(sample_costs[i])} profit: {descent.profit} learning rate: {min(system.costs)/3}")
            plt.show()
            plt.pause(0.001)'''
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
