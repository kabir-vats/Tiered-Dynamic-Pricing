import numpy as np
from scipy.stats import norm
from typing import List, Tuple

from pricing.static.system import TieredPricingSystem

class BayesianEstimator:
    """
    Maintains and updates beliefs about customer population parameters.
    Uses conjugate priors where possible for efficient updates.
    """
    def __init__(
        self,
        n_tiers: int,
        system: TieredPricingSystem,
        prior_confidence: float = 10.0
    ):
        # Initialize priors for mu and sigma (normal-inverse-gamma)
        self.beta_0 = prior_confidence
        self.alpha_0 = prior_confidence

        
        # Initialize prior for lambda (gamma distribution)
        self.lam_alpha = prior_confidence
        self.lam_beta = prior_confidence / prior_lam

        self.system = system
        
        # Maintain history of choices for updating
        self.choice_history: List[Tuple[List[float], int]] = []
        
    def update(self, prices: List[float], chosen_tier: int, utility: float) -> None:
        """
        Update parameter estimates based on a new observation.
        """
        self.choice_history.append((prices, chosen_tier))
        
        # Update lambda estimate using MAP estimation
        # Assuming customer chose optimal tier given their valuation parameter
        costs_ratio = max(prices) / min(prices)
        self.lam_est = -np.log(utility / max(prices)) / np.log(costs_ratio)
        
        # Update mu and sigma using price thresholds
        if len(self.choice_history) > 1:
            choices = np.array([c[1] for c in self.choice_history])
            # Use price ratios to estimate bounds on valuation parameter
            lower_bounds = []
            upper_bounds = []
            for p, c in self.choice_history:
                if c > 0:  # Not lowest tier
                    lower_bounds.append(p[c] / (max(prices) / min(prices))**self.lam_est)
                if c < len(p)-1:  # Not highest tier
                    upper_bounds.append(p[c+1] / (max(prices) / min(prices))**self.lam_est)
            
            if lower_bounds and upper_bounds:
                self.mu_est = np.mean(lower_bounds + upper_bounds)
                self.sigma_est = np.std(lower_bounds + upper_bounds) * 2

    def get_parameters(self) -> Tuple[float, float, float]:
        """Return current estimates of mu, sigma, and lambda."""
        return self.mu_est, self.sigma_est, self.lam_est
