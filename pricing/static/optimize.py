import numpy as np
import sys
from typing import List
from pricing.static.system import TieredPricingSystem
from scipy.optimize import dual_annealing, OptimizeResult
from scipy.stats import norm


class DualAnnealing:
    """
    Dual Annealing Optimizer for Maximizing Profit in a Tiered Pricing System.

    This class interfaces scipy's dual_annealing optimizer and serves as a
    baseline 'correct' optimized result to compare to other optimizations

    Parameters
    ----------
    system : TieredPricingSystem
        An instance of the tiered pricing system.
    """

    def __init__(self, system: TieredPricingSystem):
        self.system = system
        if system.pdf_type == "uniform":
            self.price_bounds = [
                (cost, cost * (system.mu + system.sigma)) for cost in system.costs
            ]
        else:
            self.price_bounds = [
                (cost, cost * (system.mu + system.sigma) * 3) for cost in system.costs
            ]

    def objective(self, prices: List[float]) -> float:
        """
        Compute the objective function of the system for certain prices.

        Parameters
        ----------
        prices : List[float]
            Current prices for each tier.

        Returns
        -------
        objective : float
            Negative profit of the system.
        """
        return -self.system.profit(prices)

    def maximize(self) -> OptimizeResult:
        """
        Find the optimal prices and profits for the tiered pricing system

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The optimizeresult from scipy

        Updates
        -------
        self.prices : List[float]
            Optimized prices after convergence or reaching the maximum iterations.
        self.profit : float
            Maximum profit achieved during the optimization process.
        """
        result = dual_annealing(self.objective, self.price_bounds)
        self.profit = -result.fun
        self.prices = result.x
        return result


class GradientDescent:
    """
    Gradient Descent Optimizer with Adam for Maximizing Profit.

    Implements the Adam optimization algorithm for numerical or analytic
    gradient descent to optimize a tiered pricing system. This approach
    adapts the per-parameter learning rates to accelerate convergence.

    Attributes
    ----------
    system : TieredPricingSystem
        The tiered pricing system being optimized.
    tolerance : float
        Convergence threshold for stopping.
    max_iters : int
        Maximum number of iterations allowed.
    gradient_delta : float
        Step size used for numerical gradient approximations.
    learning_rate : float
        Initial learning rate for the Adam algorithm.
    beta1 : float
        Exponential decay rate for the first moment estimates.
    beta2 : float
        Exponential decay rate for the second moment estimates.
    epsilon : float
        Small constant for numerical stability in Adam updates.
    gradient_method : str
        Method used for gradient calculation ('numerical' or 'analytic').
    """

    def __init__(
        self,
        system: TieredPricingSystem,
        tolerance: float = 1e-6,
        max_iters: int = 1000,
        gradient_delta: float = 1e-6,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        gradient_method: str = "analytic",
    ) -> None:
        self.system = system
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.gradient_delta = gradient_delta
        self.learning_rate = 3 * min(
            min(system.costs),
            lr
            * min(system.costs)
            * (max(system.costs) / min(system.costs)) ** (system.lam),
        )  # TODO: IMPROVE THIS
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gradient_method = gradient_method

    def gradient(self, prices_unsorted: List[float]) -> List[float]:
        """
        Compute the analytical gradient of the profit function.

        Parameters
        ----------
        prices_unsorted : List[float]
            Current prices for each tier in unsorted order.

        Returns
        -------
        List[float]
            The gradient vector of the profit function with respect to prices.
        """
        sorted_indices = np.argsort(self.system.utils)
        utils = np.array(self.system.utils)[sorted_indices]
        prices = np.array(prices_unsorted)[sorted_indices]
        costs = np.array(self.system.costs)[sorted_indices]

        thresholds = [-sys.float_info.max]
        t_grads = [{}]

        for i in range(self.system.tiers):
            if i == 0:
                thresholds.append(prices[0] / utils[0])
                t_grads.append({1: 1 / utils[0]})
            else:
                intersection = (prices[i] - prices[i - 1]) / (utils[i] - utils[i - 1])
                t_grad = {
                    i: -1 / (utils[i] - utils[i - 1]),
                    i + 1: 1 / (utils[i] - utils[i - 1]),
                }
                j = 0
                while intersection < thresholds[i - j]:
                    j += 1
                    if i == j:
                        intersection = prices[i] / utils[i]
                        t_grad = {i + 1: 1 / utils[i]}
                        break
                    else:
                        intersection = (prices[i] - prices[i - j - 1]) / (
                            utils[i] - utils[i - j - 1]
                        )
                        t_grad = {
                            i - j: -1 / (utils[i] - utils[i - j - 1]),
                            i + 1: 1 / (utils[i] - utils[i - j - 1]),
                        }
                thresholds.append(intersection)
                t_grads.append(t_grad)

        for i in range(self.system.tiers):
            if (
                thresholds[self.system.tiers - i]
                < thresholds[self.system.tiers - i - 1]
            ):
                thresholds[self.system.tiers - i - 1] = thresholds[
                    self.system.tiers - i
                ]
                t_grads[self.system.tiers - i - 1] = t_grads[self.system.tiers - i]

        t_grads.append({})
        intervals = [
            (thresholds[i], thresholds[i + 1]) for i in range(self.system.tiers)
        ]
        intervals.append((thresholds[-1], sys.float_info.max))

        grad = [0.0] * len(prices)

        if self.system.pdf_type == "uniform":

            start, end = (
                self.system.mu - self.system.sigma,
                self.system.mu + self.system.sigma,
            )
            point_prob = 1 / (end - start)
            for i in range(len(prices)):
                prb_grads = []
                for j in range(0, len(prices) + 1):
                    d_end = t_grads[j + 1][i + 1] if i + 1 in t_grads[j + 1] else 0
                    d_start = t_grads[j][i + 1] if i + 1 in t_grads[j] else 0
                    prb_grad = (d_end if intervals[j][1] <= end else 0) - (
                        d_start if intervals[j][0] >= start else 0
                    )
                    prb_grads.append(
                        prb_grad * point_prob
                        if (intervals[j][0] < intervals[j][1])
                        else 0
                    )
                grad[i] = sum((prices - costs) * (np.array(prb_grads[1:]))) + max(
                    (min(intervals[i + 1][1], end) - max(intervals[i + 1][0], start))
                    * point_prob,
                    0,
                )

        else:
            for i in range(len(prices)):
                prb_grads = []
                for j in range(0, len(prices) + 1):
                    d_end = t_grads[j + 1][i + 1] if i + 1 in t_grads[j + 1] else 0
                    d_start = t_grads[j][i + 1] if i + 1 in t_grads[j] else 0
                    prb_grad = d_end * norm.pdf(
                        intervals[j][1], loc=self.system.mu, scale=self.system.sigma
                    ) - d_start * norm.pdf(
                        intervals[j][0], loc=self.system.mu, scale=self.system.sigma
                    )
                    prb_grads.append(
                        prb_grad if (intervals[j][0] < intervals[j][1]) else 0
                    )
                grad[i] = sum((prices - costs) * (np.array(prb_grads[1:]))) + max(
                    norm.cdf(
                        intervals[i + 1][1], loc=self.system.mu, scale=self.system.sigma
                    )
                    - norm.cdf(
                        intervals[i + 1][0], loc=self.system.mu, scale=self.system.sigma
                    ),
                    0,
                )

        grad_ordered = [None] * len(prices)
        for i, idx in enumerate(sorted_indices):
            grad_ordered[idx] = grad[i]
        return grad_ordered

    def numerical_gradient(self, prices: List[float]) -> List[float]:
        """
        Compute the numerical gradient of the profit function.

        Parameters
        ----------
        prices : List[float]
            Current prices for each tier.

        Returns
        -------
        List[float]
            The numerical gradient vector of the profit function.
        """
        grad = [0.0] * len(prices)
        for i in range(len(prices)):
            vec = [0.0] * len(prices)
            vec[i] = 1.0
            grad[i] = (
                self.system.profit(prices + self.gradient_delta * np.asarray(vec))
                - self.system.profit(prices)
            ) / self.gradient_delta
        return grad

    def maximize(self) -> None:
        """
        Perform optimization using Adam algorithm.

        Updates
        -------
        self.prices : List[float]
            Optimized prices after convergence or reaching maximum iterations.
        self.profit : float
            Maximum profit achieved during optimization.
        self.profit_history : List[float]
            History of profit values at each iteration.
        self.price_history : List[List[float]]
            History of price vectors at each iteration.
        """
        self.prices = list(self.system.costs)
        self.profit = self.system.profit(self.prices)
        self.profit_history = [self.profit]
        self.price_history = [self.prices]

        # Initialize Adam parameters
        m_t = np.zeros(len(self.prices))  # First moment vector
        v_t = np.zeros(len(self.prices))  # Second moment vector
        t = 0  # Iteration counter

        for _ in range(self.max_iters):
            t += 1
            # print("prices " + str(self.prices))
            grad = (
                np.array(self.gradient(self.prices))
                if self.gradient_method == "analytic"
                else np.array(self.numerical_gradient(self.prices))
            )

            # Update moments
            m_t = self.beta1 * m_t + (1 - self.beta1) * grad
            v_t = self.beta2 * v_t + (1 - self.beta2) * (grad**2)

            # Bias correction
            m_t_hat = m_t / (1 - self.beta1**t)
            v_t_hat = v_t / (1 - self.beta2**t)

            # Update prices
            prices_next = np.array(self.prices) + (
                self.learning_rate * m_t_hat / (np.sqrt(v_t_hat) + self.epsilon)
            )

            # Early stopping if change in prices is below tolerance
            if np.linalg.norm(prices_next - self.prices) < self.tolerance:
                break

            self.prices = prices_next.tolist()
            self.profit = self.system.profit(self.prices)
            self.profit_history.append(self.profit)
            self.price_history.append(self.prices.copy())
