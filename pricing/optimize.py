import numpy as np
from typing import List
from pricing.system import TieredPricingSystem
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
        self.prices = -result.x
        return result
