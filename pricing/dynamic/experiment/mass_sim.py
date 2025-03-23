from pricing.dynamic.customer import Customer
from pricing.dynamic.business import Business
from pricing.dynamic.controller import StochasticGradientDescent
from pricing.static.optimize import DualAnnealing
from pricing.static.system import TieredPricingSystem
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compare_convergence(C, lam, mu, sigma, pdf_type):
    """
    Compare the convergence of two optimization methods: Stochastic Gradient Descent and Dual Annealing.

    Parameters:
    C (list): List of cost values.
    lam (float): Lambda parameter for the customer.
    mu (float): Mean of the normal distribution for the customer.
    sigma (float): Standard deviation of the normal distribution for the customer.
    pdf_type (str): Type of probability density function.

    Returns:
    tuple: A tuple containing the profits obtained from the Stochastic Gradient Descent and Dual Annealing methods.
    """
    customer = Customer(mu, sigma, lam)
    business = Business(C, customer)
    controller = StochasticGradientDescent(business, max_iters=300, lr=0.1, batch_size=10)
    system = TieredPricingSystem(C, len(C), lam, mu, sigma)
    dual = DualAnnealing(system)
    controller.maximize()
    dual.maximize()
    return (system.profit(controller.prices), system.profit(dual.prices))


def main():
    profits = []
    for i in tqdm(range(10)):
        for j in range(i+2, 10):
            for k in range(j+1, 10):
                for l in range(10):
                    C = [i+1, j, k]
                    lam = l / 10
                    mu = 8
                    sigma = 1
                    pdf_type = "gaussian"
                    profits.append(compare_convergence(C, lam, mu, sigma, pdf_type))

    # Convert the list of tuples to a NumPy array for easier manipulation
    profits_array = np.array(profits)

    # Extract the profits from Stochastic Gradient Descent and Dual Annealing
    sgd_profits = profits_array[:, 0]
    dual_profits = profits_array[:, 1]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define the bar width
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    sgd_positions = np.arange(len(sgd_profits))
    dual_positions = sgd_positions + bar_width

    # Create the bars
    ax.bar(sgd_positions, sgd_profits, width=bar_width, label='Stochastic Gradient Descent')
    ax.bar(dual_positions, dual_profits, width=bar_width, label='Dual Annealing')

    # Add labels and title
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Profit')
    ax.set_title('Comparison of Stochastic Gradient Descent and Dual Annealing Profits')
    ax.set_xticks(sgd_positions + bar_width / 2)
    ax.set_xticklabels(np.arange(len(sgd_profits)))  # You can replace this with more descriptive labels if needed
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    


if __name__ == "__main__":
    main()