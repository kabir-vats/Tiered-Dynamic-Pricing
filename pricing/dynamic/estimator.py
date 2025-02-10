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
        system: TieredPricingSystem,
        a_prior: Tuple[float, float] = (2, 2),
        b_prior: Tuple[float, float] = (2, 2),
        lam_prior: Tuple[float, float] = (0.5, 0.5),
        num_samples: int = 100,
    ):
        self.a_mean, self.a_std = a_prior
        self.b_mean, self.b_std = b_prior
        self.lambda_mean, self.lambda_std = lam_prior
        self.num_samples = num_samples
        self.system = system

        self.a_posterior = [self.a_mean]
        self.b_posterior = [self.b_mean]
        self.lambda_posterior = [self.lambda_mean]
        
    def update(self, prices: List[float], choices: List[int]) -> None:
        """
        Update parameter estimates based on a new observation.
        """
        # Sample parameter values from prior distributions
        a_samples = np.random.uniform(self.a_mean-self.a_std, self.a_mean+self.a_std, self.num_samples)
        b_samples = np.random.uniform(self.b_mean-self.b_std, self.a_mean+self.b_std, self.num_samples)
        lambda_samples = np.random.uniform(self.lambda_mean-self.lambda_std, self.lambda_mean+self.lambda_std, self.num_samples)
        
        # Ensure valid values
        a_samples = np.clip(a_samples, 0, np.inf)
        b_samples = np.clip(b_samples, 0, np.inf)
        lambda_samples = np.clip(lambda_samples, 0, 1)
        a_sam = []
        b_sam = []
        lam_sam = []
        for a,b,lam in zip(a_samples, b_samples, lambda_samples):
            if (b>a):
                a_sam.append(a)
                b_sam.append(b)
                lam_sam.append(lam)

        
        # Compute likelihoods for each sample
        likelihoods = []

        for a, b, lam in zip(a_sam, b_sam, lam_sam):
            if (b<a):
                continue
            self.system.update_parameters((a+b)/2, (b-a)/2, lam)
            probs = self.system.tier_probabilities(prices)
            conditional_prob = 1
            for choice in choices:
                conditional_prob *= probs[choice]
            if (conditional_prob > 0):
                print(a,b,lam)
                print(conditional_prob)
                self.system.update_parameters(1, 3, 2/3)
                probs = self.system.tier_probabilities(prices)
                conditional_prob = 1
                for choice in choices:
                    conditional_prob *= probs[choice]
                print("real ", conditional_prob)
            likelihoods.append(conditional_prob)

        likelihoods = np.array(likelihoods)
        likelihoods = likelihoods.astype(np.float64)
        likelihoods += 1e-8  # Avoid zero probabilities
        

        # Compute posterior weights
        posterior_weights = likelihoods / np.sum(likelihoods)
        
        self.a_posterior.append(np.sum(posterior_weights * a_sam))
        self.b_posterior.append(np.sum(posterior_weights * b_sam))
        self.lambda_posterior.append(np.sum(posterior_weights * lam_sam))
        
        self.a_mean = np.mean(self.a_posterior[-10:])
        self.b_mean = np.mean(self.b_posterior[-10:])
        self.lambda_mean = np.mean(self.lambda_posterior[-10:])

        self.system.update_parameters((self.a_mean+self.b_mean)/2, (self.a_mean-self.b_mean)/2, self.lambda_mean)
        
        # Reduce uncertainty over time
        '''self.a_std = max(self.a_std * 0.99, 0.1)
        self.b_std = max(self.b_std * 0.99, 0.1)
        self.lambda_std = max(self.lambda_std * 0.99, 0.05)'''
