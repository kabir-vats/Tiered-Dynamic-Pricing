import numpy as np
from tqdm import tqdm
from pricing.static.system import TieredPricingSystem


class BatchGradientDescent:
    def __init__(self, business, batch_size: int = 100,
                 max_iters: int = 1500, gradient_delta: float = 1e-1,
                 lr: int = None, beta1=0.9, beta2=0.999, epsilon=1e-8, smoothing_alpha=0) -> None:
        self.business = business
        self.system = TieredPricingSystem(business.costs, len(business.costs),
                                          business.customer.scaling_param, business.customer.mu,
                                          business.customer.sigma)
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
            self.learning_rate = min(business.costs) / 10

    def estimate_gradient(self):
        # Use partial or stale data
        grad = [0.0] * len(self.prices)
        for i in range(len(self.prices)):
            vec = [0.0] * len(self.prices)
            vec[i] = 1.0
            # Sample fewer data points
            grad[i] = (
                self.business.sell_n(self.prices + self.gradient_delta
                                     * np.asarray(vec), self.batch_size)[0]
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
            v_t = self.beta2 * v_t + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_t_hat = m_t / (1 - self.beta1 ** t)
            v_t_hat = v_t / (1 - self.beta2 ** t)

            prices_next = np.array(self.prices) + (self.learning_rate * m_t_hat /
                                                   (np.sqrt(v_t_hat) + self.epsilon))

            self.prices = prices_next.tolist()
            self.profit = self.system.profit(self.prices)
            self.profit_history.append(self.profit)
            self.price_history.append(self.prices.copy())
            self.learning_rate *= 0.99
        print(self.business.transaction_history)
