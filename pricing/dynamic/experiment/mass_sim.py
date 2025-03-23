from pricing.dynamic.customer import Customer
from pricing.dynamic.business import Business
from pricing.dynamic.controller import StochasticGradientDescent
from pricing.static.optimize import DualAnnealing
from pricing.static.system import TieredPricingSystem


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
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for l in range(10):
                    C = [i, j, k]
                    lam = l / 10
                    mu = 8
                    sigma = 1
                    pdf_type = "gaussian"
                    profits.append(compare_convergence(C, lam, mu, sigma, pdf_type))
