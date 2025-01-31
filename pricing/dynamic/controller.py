from typing import List
import numpy as np
from tqdm import tqdm
from pricing.static.system import TieredPricingSystem


class BatchGradientDescent:
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
        batch_size: int = 1000,
        max_iters: int = 1500,
        gradient_delta: float = 1e-1,
        lr: int = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        smoothing_alpha: float = 0,
    ) -> None:
        self.business = business
        self.system = TieredPricingSystem(
            business.costs,
            len(business.costs),
            business.customer.lam,
            business.customer.mu,
            business.customer.sigma,
        )
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

    def estimate_gradient(self) -> List[float]:
        """
        Estimate the gradient of the profit function using batch samples.

        Returns
        -------
        List[float]
            The (possibly smoothed) gradient vector for each price tier.

        Updates
        -------
        self.smoothed_grad : List[float]
            Stores the updated smoothed gradient vector.
        """
        # Use partial or stale data
        grad = [0.0] * len(self.prices)
        for i in range(len(self.prices)):
            vec = [0.0] * len(self.prices)
            vec[i] = 1.0
            # Sample fewer data points
            grad[i] = (
                self.business.sell_n(
                    self.prices + self.gradient_delta * np.asarray(vec), self.batch_size
                )[0]
                - self.business.sell_n(self.prices, self.batch_size)[0]
            ) / self.gradient_delta
        if self.smoothed_grad is None:
            self.smoothed_grad = [g * (1 - self.smoothing_alpha) for g in grad]
        else:
            for i in range(len(grad)):
                self.smoothed_grad[i] = (
                    self.smoothing_alpha * self.smoothed_grad[i]
                    + (1 - self.smoothing_alpha) * grad[i]
                )
        return self.smoothed_grad

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
        self.prices = list(self.business.costs)
        self.profit = self.business.sell_n(self.prices, self.batch_size)[0]
        self.profit_history = [self.profit]
        self.price_history = [self.prices]

        m_t = np.zeros(len(self.prices))  # First moment vector
        v_t = np.zeros(len(self.prices))  # Second moment vector
        t = 0  # Iteration counter

        for i in tqdm(range(self.max_iters)):
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

            self.prices = prices_next.tolist()
            self.profit = self.system.profit(self.prices)
            self.profit_history.append(self.profit)
            self.price_history.append(self.prices.copy())
        print(self.business.transaction_history)
