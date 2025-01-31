import numpy as np
import sys
from typing import List
from pricing.static.system import TieredPricingSystem
from scipy.optimize import dual_annealing, OptimizeResult


class GradientDescent:
    """
    Gradient Descent Optimizer for Maximizing Profit in a Tiered Pricing System.

    This class implements a gradient descent algorithm to optimize the pricing
    strategy for a given system by approximating gradients numerically.

    Parameters
    ----------
    system : TieredPricingSystem
        An instance of the tiered pricing system.
    tolerance : float, optional
        Convergence tolerance for early stopping. The algorithm stops when the
        change in prices is smaller than this value. Default is 1e-6.
    max_iters : int, optional
        Maximum number of iterations for the optimization process. Default is 1000.
    gradient_delta : float, optional
        Small change in price used to approximate gradients numerically.
        Default is 1e-6.
    patience : int, optional
        Number of consecutive iterations with decreasing profit before halving
        the learning rate. Default is 1.
    """

    def __init__(self, system: TieredPricingSystem, tolerance: float = 1e-6,
                 max_iters: int = 1000, gradient_delta: float = 1e-6,
                 lr: int = None, patience: int = 1) -> None:
        self.system = system
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.gradient_delta = gradient_delta
        self.patience = patience
        self.learning_rate = lr
        if lr is None:
            self.learning_rate = min(system.costs) / 5

    def numerical_gradient(self, prices: List[float]) -> List[float]:
        """
        Compute the numerical gradient of the profit function with respect to prices.

        Parameters
        ----------
        prices : List[float]
            Current prices for each tier.

        Returns
        -------
        grad : List[float]
            Numerical gradient of the profit function.
        """
        grad = [0.0] * len(prices)
        for i in range(len(prices)):
            vec = [0.0] * len(prices)
            vec[i] = 1.0
            grad[i] = (
                self.system.profit(prices + self.gradient_delta * np.asarray(vec)) -
                self.system.profit(prices)
            ) / self.gradient_delta
        return grad

    def maximize(self) -> None:
        """
        Perform gradient descent to maximize profit.

        This method iteratively updates the prices to maximize the profit based
        on the numerical gradient of the profit function.

        Updates
        -------
        self.prices : List[float]
            Optimized prices after convergence or reaching the maximum iterations.
        self.profit : float
            Maximum profit achieved during the optimization process.
        self.profit_history : List[float]
            History of profit values at each iteration.
        self.price_history : List[List[float]]
            History of price vectors at each iteration.
        """
        self.prices = list(self.system.costs)
        self.profit = self.system.profit(self.prices)
        self.profit_history = [self.profit]
        self.price_history = [self.prices]
        patience_count = 0

        for _ in range(self.max_iters):
            grad = self.numerical_gradient(self.prices)
            prices_next = self.prices + np.asarray(self.learning_rate) * grad

            # Early stopping if change in prices is below tolerance
            if np.linalg.norm(prices_next - self.prices) < self.tolerance:
                break

            self.prices = prices_next
            self.profit = self.system.profit(prices_next)
            self.profit_history.append(self.profit)
            self.price_history.append(prices_next.copy())

            # Adjust learning rate if profit decreases
            if self.profit_history[-1] < self.profit_history[-2]:
                patience_count += 1
                if patience_count == self.patience:
                    self.learning_rate *= 0.5
                    patience_count = 0


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
        if (system.pdf_type == 'uniform'):
            self.price_bounds = [(cost, cost * (system.mu + system.sigma))
                                 for cost in system.costs]
        else:
            self.price_bounds = [(cost, cost * (system.mu + system.sigma) * 3)
                                 for cost in system.costs]

    def objective(self, prices: List[(float)]) -> float:
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


class GradientDescentAdam:
    """
    Gradient Descent Optimizer with Adam for Maximizing Profit.

    Implements the Adam optimization algorithm for numerical gradient descent
    to optimize tiered pricing.
    """
    def __init__(self, system, tolerance=1e-6, max_iters=1000, gradient_delta=1e-6,
                 lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, gradient_method='numerical'):
        self.system = system
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.gradient_delta = gradient_delta
        self.learning_rate = min(min(system.costs), lr * min(system.costs) *
                                 (max(system.costs) / min(system.costs)) **
                                 (system.lam))  # TODO: Better expressio
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.gradient_method = gradient_method

    def gradient(self, prices_unsorted):
        """Compute the gradient of the profit function."""
        sorted_indices = np.argsort(self.system.utils)
        utils = np.array(self.system.utils)[sorted_indices]
        prices = np.array(prices_unsorted)[sorted_indices]
        costs = np.array(self.system.costs)[sorted_indices]

        thresholds = [-sys.float_info.max]
        t_grads = [{}]

        for i in range(self.system.tiers):
            if i == 0:
                thresholds.append(prices[0] / utils[0])
                t_grads.append({1: 1/utils[0]})
            else:
                intersection = (prices[i] - prices[i - 1]) / (
                    utils[i] - utils[i-1]
                )
                t_grad = {i: -1 / (utils[i] - utils[i-1]), i+1: 1 / (utils[i] - utils[i-1])}
                j = 0
                while intersection < thresholds[i - j]:
                    j += 1
                    if i == j:
                        intersection = prices[i] / utils[i]
                        t_grad = {i+1: 1 / utils[i]}
                        break
                    else:
                        intersection = (prices[i] - prices[i - j - 1]) / (
                            utils[i] - utils[i-j-1]
                        )
                        t_grad = {i-j: -1 / (utils[i] - utils[i-j-1]), i+1: 1 / (utils[i] - utils[i-j-1])}
                thresholds.append(intersection)
                t_grads.append(t_grad)

        for i in range(self.system.tiers):
            if thresholds[self.system.tiers - i] < thresholds[self.system.tiers - i - 1]:
                thresholds[self.system.tiers - i - 1] = thresholds[self.system.tiers - i]
                t_grads[self.system.tiers - i - 1] = t_grads[self.system.tiers - i]

        t_grads.append({})
        intervals = [(thresholds[i], thresholds[i + 1]) for i in range(self.system.tiers)]
        intervals.append((thresholds[-1], sys.float_info.max))

        grad = [0.0] * len(prices)
        print(t_grads)

        start, end = self.system.mu - self.system.sigma, self.system.mu + self.system.sigma
        point_prob = 1 / (end - start)
        for i in range(len(prices)):
            prb_grads = []
            for j in range(0, len(prices) + 1):
                d_end = (t_grads[j+1][i+1] if i+1 in t_grads[j+1] else 0)
                d_start = (t_grads[j][i+1] if i+1 in t_grads[j] else 0)
                #print(d_start)
                #print(d_end)
                #print(intervals)
                prb_grad = (d_end if intervals[j][1] <= end else 0) - (d_start if intervals[j][0] >= start else 0)
                prb_grads.append(prb_grad * point_prob if (intervals[j][0] < intervals[j][1]) else 0)
            print(prb_grads)
            grad[i] = sum((prices - costs) * (np.array(prb_grads[1:]))) + max((min(intervals[i+1][1], end) - max(intervals[i+1][0], start))
                                    * point_prob, 0)
            print(sum((prices - costs) * (np.array(prb_grads[1:]))))
        #print(t_grads)
        #print(grad)
        return grad

    def numerical_gradient(self, prices):
        """Compute the numerical gradient of the profit function."""
        grad = [0.0] * len(prices)
        for i in range(len(prices)):
            vec = [0.0] * len(prices)
            vec[i] = 1.0
            grad[i] = (
                self.system.profit(prices + self.gradient_delta * np.asarray(vec)) -
                self.system.profit(prices)
            ) / self.gradient_delta
        return grad

    def maximize(self):
        """
        Perform optimization using Adam.
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
            print("prices " + str(self.prices))
            grad = np.array(self.gradient(self.prices)) if self.gradient_method == 'analytic' else np.array(self.numerical_gradient(self.prices))

            # Update moments
            m_t = self.beta1 * m_t + (1 - self.beta1) * grad
            v_t = self.beta2 * v_t + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_t_hat = m_t / (1 - self.beta1 ** t)
            v_t_hat = v_t / (1 - self.beta2 ** t)

            # Update prices
            prices_next = np.array(self.prices) + (self.learning_rate * m_t_hat /
                                                   (np.sqrt(v_t_hat) + self.epsilon))

            # Early stopping if change in prices is below tolerance
            if np.linalg.norm(prices_next - self.prices) < self.tolerance:
                break

            self.prices = prices_next.tolist()
            self.profit = self.system.profit(self.prices)
            self.profit_history.append(self.profit)
            self.price_history.append(self.prices.copy())
