from typing import List
import numpy as np
from tqdm import tqdm
from pricing.static.optimize import GradientDescent
from pricing.static.system import TieredPricingSystem
from pricing.dynamic.estimator import BayesianEstimator


class StochasticGradientDescent:
    """
    Class for performing stochastic gradient descent with batches

    Parameters
    ----------
    business : Any
        An object representing the business containing costs and a customer.
    batch_size : int
        Number of samples per batch for gradient estimation.
    max_iters : int
        Maximum number of iterations for gradient descent.
    gradient_delta : float
        Delta used for numerical gradient approximation.
    lr : int, optional
        Initial learning rate value.
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Small constant for numerical stability in Adam-like updates.
    smoothing_alpha : float
        Smoothing factor for the estimated gradient.
    """

    def __init__(
        self,
        business,
        batch_size: int = 1,
        max_iters: int = 200,
        gradient_delta: float = 1e-1,
        lr: int = 0.03,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        smoothing_alpha: float = 0,
        pdf_type: str = "uniform",
    ) -> None:
        self.business = business
        self.max_iters = max_iters
        self.gradient_delta = gradient_delta
        self.batch_size = batch_size
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.smoothing_alpha = smoothing_alpha
        self.smoothed_grad = None
        if lr is None:
            self.learning_rate = min(business.costs) / 5

        self.mock_system = TieredPricingSystem(
            self.business.costs, len(self.business.costs), 1, 1, 1, pdf_type=pdf_type
        )  # dummy values for mu, sigma, lambda

        self.estimator = BayesianEstimator.get(pdf_type, self.mock_system, 100000)
        self.mock_descent = GradientDescent(self.mock_system)

    def estimate_gradient(self) -> List[float]:
        """
        Estimate gradient using both batch samples and analytical computation
        with estimated parameters.
        """
        # Get batch of customer choices
        profits = []
        choices = []
        '''for _ in range(self.batch_size):
            profit, choice = self.business.sell(self.prices)
            profits.append(profit)
            choices.append(choice)'''
        

        # Use jitter to sell at prices near or far from est. prices
        jitter_factor = 0.5
        jittered_prices = [p * (1 + jitter_factor * (2 * np.random.random() - 1)) 
                            for p in self.prices]
        jittered_prices = [max(cost, price) for cost, price in zip(self.business.costs, jittered_prices)]
    
        for _ in range(self.batch_size):
            profit, choice = self.business.sell(jittered_prices)
            profits.append(profit)
            choices.append(choice)

        self.estimator.update(jittered_prices, choices)

        # self.profit_history.append(np.mean(profits))
        '''if self.iters % 10 == 0:
            print(Counter(choices))
            print(self.prices)
            print(self.estimator.a_mean)
            print(self.estimator.b_mean)
            print(self.estimator.lambda_mean)
            # print(self.estimator.likelihood_posterior)
            # print(self.estimator.a_posterior)
            # print(self.estimator.b_posterior)
            """mu = (self.estimator.a_mean + self.estimator.b_mean) / 2
            sigma = (self.estimator.b_mean - self.estimator.a_mean) / 2
            self.mock_system.update_parameters(mu, sigma, self.estimator.lambda_mean)
            print(self.mock_system.tier_probabilities(self.prices))
            self.mock_system.update_parameters(2, 1, 2/3)
            print(self.mock_system.tier_probabilities(self.prices))"""
            input("cont")'''

        return self.mock_descent.gradient(self.prices)

        # Combine with empirical gradient for robustness
        """empirical_grad = self._empirical_gradient(profits, choices)
        combined_grad = [
            0.7 * ag + 0.3 * eg
            for ag, eg in zip(analytical_grad, empirical_grad)
        ]

        return combined_grad"""

    def _empirical_gradient(
        self, profits: List[float], choices: List[int]
    ) -> List[float]:
        """Compute empirical gradient from batch samples."""
        grad = [0.0] * len(self.prices)
        for i in range(len(self.prices)):
            vec = [0.0] * len(self.prices)
            vec[i] = 1.0
            grad[i] = (
                np.mean(profits) - self.business.sell_n(self.prices, self.batch_size)[0]
            ) / self.gradient_delta
        return grad

    def maximize(self) -> None:
        """
        Run the batch gradient descent process to maximize profit.

        Updates
        -------
        self.prices : List[float]
            Optimized prices over iterations.
        self.profit : float
            Final profit result.
        self.profit_history : List[float]
            Record of profit at each iteration.
        self.price_history : List[List[float]]
            Record of prices at each iteration.
        """
        self.prices = self.business.costs
        self.profit = self.business.sell_n(self.prices, self.batch_size)[0]
        self.profit_history = [self.profit]
        self.price_history = [self.prices]

        m_t = np.zeros(len(self.prices))   # First moment vector
        v_t = np.zeros(len(self.prices))  # Second moment vector
        t = 0  # Iteration counter

        for i in tqdm(range(self.max_iters)):
            self.iters = i
            grad = np.array(self.estimate_gradient())
            t += 1
            
            # Update moments
            m_t = self.beta1 * m_t + (1 - self.beta1) * grad
            v_t = self.beta2 * v_t + (1 - self.beta2) * (grad**2)

            # Bias correction
            m_t_hat = m_t / (1 - self.beta1**t)
            v_t_hat = v_t / (1 - self.beta2**t)

            prices_next = np.array(self.prices) + (
                self.learning_rate * m_t_hat / (np.sqrt(v_t_hat) + self.epsilon)
            )

            self.prices = prices_next.copy()
            self.profit = self.mock_system.profit(self.prices)
            self.profit_history.append(self.profit)
            self.price_history.append(self.prices.copy())
