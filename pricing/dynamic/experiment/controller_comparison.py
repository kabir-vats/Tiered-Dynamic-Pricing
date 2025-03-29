from pricing.dynamic.controller import StochasticGradientDescent
from pricing.dynamic.customer import Customer
from pricing.dynamic.business import Business
from pricing.static.system import TieredPricingSystem
from pricing.static.optimize import DualAnnealing, GradientDescent
from pricing.util.visualize import (
    descent_label_lr,
    plot_descent_two_tiers,
    plot_descent_one_tier,
    compare_descents_two_tiers,
    compare_n_descents_two_tiers,
    descent_title,
    plot_descent_three_tiers,
    compare_descents_three_tiers,
    descent_label_lr_profit,
    plot_choice_distribution,
    plot_parameter_history, 
    compare_parameter_history,
    compare_profit_history,
    compare_convergence_metrics,
)
from pricing.util.simulate import simulate_profits
import matplotlib.pyplot as plt
import numpy as np


def compare_pricing_three_tiers():
    np.set_printoptions(legacy="1.25")
    C = [1, 5, 20]
    lambda_value = 2/3
    mu = 3
    sigma = 1
    max_iters = 800
    learning_rates = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    # TRIAL SETTINGS
    max_iters = 800
    # learning_rates = [0.05, 0.05, 0.05]
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma, pdf_type="gaussian")
    customer = Customer(mu, sigma, lambda_value, pdf_type="gaussian")
    business = Business(C, customer)
    descents = []
    estimators = []
    smaller_labels = []
    labels = []
    for i, lr in enumerate(learning_rates):
        jitter_factor = 0.1 * (i > 4)
        descent = StochasticGradientDescent(business, batch_size=1, max_iters=max_iters, pdf_type="gaussian", lr=lr, jitter_factor=jitter_factor)
        descent.maximize()
        descents.append(descent)
        labels.append(f'Descent: {i} ' + descent_label_lr_profit(lr, system.profit(descent.prices))+ f' jitter: {jitter_factor}')
        estimators.append(descent.estimator)
        smaller_labels.append(f'Descent: {i}')

    colors = ["#3264a8", "#323aa8", "#3268a8","#323aa8", "#3268a8", "#a8326f", "#a83255", "#a83100", "#a83255", "#a83100"]
    # colors = None
    dual = DualAnnealing(system)
    dual.maximize()
    print(dual.profit)

    print(system.tier_probabilities(dual.prices))

    title = descent_title(C, lambda_value, dual.profit, "gaussian", mu, sigma) + f' Iterations: {max_iters}'

    

    fig = compare_profit_history(
        controllers=descents,
        controller_labels=labels,
        system=system,
        optimal_profit=dual.profit,
        colors=colors,
        title=f'Profit History Comparison: {title}',
        window_size=5  # Apply smoothing with window of 5
    )
    plt.show()

    fig1 = compare_parameter_history(estimators, labels, true_params=[mu, sigma, lambda_value], title=f'Parameter History Comparison: {title}', colors=colors)
    plt.show()
    
    fig = compare_descents_three_tiers(descents, labels, dual.prices, dual.profit, f'Price History Comparison: {title}', colors=colors)
    plt.show()


def test_lr_three_tiers():
    C = [1, 5, 30]
    lambda_value = 2 / 3
    mu = 3
    sigma = 1
    learning_rates = [0.01, 0.001]
    customer = Customer(mu, sigma, lambda_value, pdf_type="gaussian")
    business = Business(C, customer)
    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma, pdf_type="gaussian")
    
    descents = []
    labels = []
    for lr in learning_rates:
        descent = GradientDescent(system, lr=lr)
        descent.maximize()
        descents.append(descent)
        labels.append(descent_label_lr_profit(lr, descent.profit))

    dual = DualAnnealing(system)
    dual.maximize()
    print(dual.profit)

    print(system.tier_probabilities(descents[0].prices))

    title = descent_title(C, lambda_value, dual.profit, "gaussian", mu, sigma)
    fig = compare_descents_three_tiers(descents, labels, dual.prices, title)
    plt.show()

def compare_pricing_two_tiers():
    C = [2,5]
    lambda_value = 2 / 3
    mu = 2
    sigma = 1
    learning_rates = [0.01, 0.03, 0.05]

    customer = Customer(mu, sigma, lambda_value, pdf_type="gaussian")
    business = Business(C, customer)
    controllers = []

    system = TieredPricingSystem(C, len(C), lambda_value, mu, sigma, pdf_type="gaussian")

    for lr in learning_rates:
        controller = StochasticGradientDescent(business, max_iters=500, lr=lr, batch_size=1, pdf_type="gaussian", jitter_factor=0.5)
        controller.maximize()
        controllers.append(controller)

    profits, samples = simulate_profits(system, n_samples=100)

    labels = [descent_label_lr(lr) for lr in learning_rates]
    title = descent_title(C, lambda_value, max([descent.profit for descent in controllers]), "gaussian", mu, sigma)
    compare_n_descents_two_tiers(samples[0], samples[1], profits, controllers, labels, title)
    plt.show()


if __name__ == "__main__":
    compare_pricing_three_tiers()
    input('ok')