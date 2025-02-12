import math
import numpy as np
from scipy.stats import norm
from typing import List, Tuple
from collections import Counter

from pricing.static.system import TieredPricingSystem

class Trial:
    def __init__(self, prices: List[float], choices: List[int]):
        self.prices = prices
        self.choices = choices
        self.counts = Counter(choices)


class BayesianEstimator:
    """
    Maintains and updates beliefs about customer population parameters.
    Uses conjugate priors where possible for efficient updates.
    """
    def __init__(
        self,
        system: TieredPricingSystem,
        a_prior: Tuple[float, float] = (1.9, 5),
        b_prior: Tuple[float, float] = (2, 5),
        lam_prior: Tuple[float, float] = (0.67, 0.4),
        num_samples: int = 1000,
    ):
        self.a_mean, self.a_std = a_prior
        self.b_mean, self.b_std = b_prior
        self.lambda_mean, self.lambda_std = lam_prior
        self.num_samples = num_samples
        self.system = system
        self.prev_trials = []

        self.a_posterior = [self.a_mean]
        self.b_posterior = [self.b_mean]
        self.lambda_posterior = [self.lambda_mean]
        self.likelihood_posterior = [1]

    def param_probability(self, a: float, b:float, lam:float, trial: Trial) -> float:
        self.system.update_parameters((a+b)/2, (b-a)/2, lam)
        probs = self.system.tier_probabilities(trial.prices)
        denom = 1
        num = math.factorial(len(trial.choices))
        for i in range(len(probs)):
            denom *= math.factorial(trial.counts[i])
            num *= probs[i]**trial.counts[i]
        return num / denom
        
    def update(self, prices: List[float], choices: List[int]) -> None:
        """
        Update parameter estimates based on a new observation.
        """
        curr_trial = Trial(prices, choices)
        # Update previous probabilities based on new data
        for i in range(len(self.likelihood_posterior)):
            if self.likelihood_posterior[i] == 0:
                continue
            self.likelihood_posterior[i] *= self.param_probability(self.a_posterior[i], self.b_posterior[i], 
                                                                   self.lambda_posterior[i], curr_trial)
        index = np.argmax(self.likelihood_posterior)
        '''print('\n')
        print(self.likelihood_posterior[0])
        print(self.likelihood_posterior[index])
        print(self.a_posterior[index], self.b_posterior[index], self.lambda_posterior[index])
'''
        # Sample parameter values from prior distributions
        a_samples = np.random.uniform(self.a_mean-self.a_std, self.a_mean+self.a_std, self.num_samples)
        b_samples = np.random.uniform(self.b_mean-self.b_std, self.a_mean+self.b_std, self.num_samples)
        lambda_samples = np.random.uniform(self.lambda_mean-self.lambda_std, self.lambda_mean+self.lambda_std, self.num_samples)
        
        # Ensure valid values
        a_samples = a_samples[a_samples >= 0]
        b_samples = b_samples[b_samples >= 0]
        lambda_samples = lambda_samples[(lambda_samples >= 0) & (lambda_samples <= 0.9)]
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
        valid_choices = 0

        for a, b, lam in zip(a_sam, b_sam, lam_sam):
            conditional_prob = self.param_probability(a, b, lam, curr_trial)
            for trial in self.prev_trials:
                if conditional_prob == 0:
                    break
                conditional_prob *= self.param_probability(a, b, lam, trial)
            if (conditional_prob > 0):
                valid_choices += 1
            likelihoods.append(conditional_prob)

        # print(likelihoods)
        if valid_choices == 0:
            self.a_std *= 1.1
            self.b_std *= 1.1
            return
        '''else:
            self.a_std = max(self.a_std * 0.99, 0.1)
            self.b_std = max(self.b_std * 0.99, 0.1)'''

        likelihoods = np.array(likelihoods)
        likelihoods = likelihoods.astype(np.float64)
        

        # Compute posterior weights
        posterior_weights = likelihoods / np.sum(likelihoods)
        self.prev_trials.append(curr_trial)
        
        self.likelihood_posterior.append(np.sum(likelihoods) / valid_choices)
        self.a_posterior.append(np.sum(posterior_weights * a_sam))
        self.b_posterior.append(np.sum(posterior_weights * b_sam))
        self.lambda_posterior.append(np.sum(posterior_weights * lam_sam))

        self.a_mean = (np.sum(np.float64(self.a_posterior[-100:]) * np.float64(self.likelihood_posterior[-100:])) / np.sum(np.float64(self.likelihood_posterior[-100:])))
        self.b_mean =  (np.sum(np.float64(self.b_posterior[-100:]) * np.float64(self.likelihood_posterior[-100:])) / np.sum(np.float64(self.likelihood_posterior[-100:])))
        self.lambda_mean = (np.sum(np.float64(self.lambda_posterior[-100:]) * np.float64(self.likelihood_posterior[-100:])) / np.sum(np.float64(self.likelihood_posterior[-100:])))

        self.system.update_parameters((self.a_mean+self.b_mean)/2, (self.a_mean-self.b_mean)/2, self.lambda_mean)
        
        # Reduce uncertainty over time
        '''self.a_std = max(self.a_std * 0.99, 0.1)
        self.b_std = max(self.b_std * 0.99, 0.1)
        self.lambda_std = max(self.lambda_std * 0.99, 0.05)'''