import math
import numpy as np
from typing import List, Tuple
from collections import Counter
from pricing.static.system import TieredPricingSystem


class Trial:
    """
    Container for a single pricing trial's data.

    Parameters
    ----------
    prices : List[float]
        The prices used in this trial
    choices : List[int]
        The choices made by customers in this trial
    """

    def __init__(self, prices: List[float], choices: List[int]):
        self.prices = prices
        self.choices = choices
        self.counts = Counter(choices)


class BayesianEstimator:
    """
    Maintains and updates beliefs about customer population parameters.
    Uses conjugate priors where possible for efficient updates.

    Parameters
    ----------
    system : TieredPricingSystem
        The pricing system model to use for probability calculations
    a_prior : Tuple[float, float]
        Mean and standard deviation for the lower bound parameter prior
    b_prior : Tuple[float, float]
        Mean and standard deviation for the upper bound parameter prior
    lam_prior : Tuple[float, float]
        Mean and standard deviation for the lambda parameter prior
    num_samples : int
        Number of samples to use in parameter estimation
    """

    def __init__(
        self,
        system: TieredPricingSystem,
        a_prior: Tuple[float, float] = (-5, 5),
        b_prior: Tuple[float, float] = (0, 10),
        lam_prior: Tuple[float, float] = (0, 1),
        num_samples: int = 10000,
    ):
        self.particles = np.zeros((num_samples, 3))
        self.particles[:, 0] = np.random.uniform(a_prior[0], a_prior[1], num_samples)
        self.particles[:, 1] = np.random.uniform(b_prior[0], b_prior[1], num_samples)
        self.particles[:, 2] = np.random.uniform(
            lam_prior[0], lam_prior[1], num_samples
        )

        valid = self.particles[:, 1] > self.particles[:, 0]
        self.particles = self.particles[valid]

        self.num_samples = len(self.particles)
        self.system = system
        self.prev_trials = []

        self.weights = np.ones(self.num_samples) / self.num_samples

    def param_probability(self, a: float, b: float, lam: float, trial: Trial) -> float:
        """
        Calculate the probability of observing a trial's outcomes given parameters.

        Parameters
        ----------
        a : float
            Lower bound parameter
        b : float
            Upper bound parameter
        lam : float
            Lambda parameter
        trial : Trial
            The trial data to evaluate

        Returns
        -------
        float
            The probability of the trial outcomes given the parameters
        """
        self.system.update_parameters((a + b) / 2, (b - a) / 2, lam)
        probs = self.system.tier_probabilities(trial.prices)
        prob = math.factorial(len(trial.choices))
        for i in range(len(probs)):
            prob /= math.factorial(trial.counts[i])
            prob *= probs[i] ** trial.counts[i]
        return prob

    def update(self, prices: List[float], choices: List[int]) -> None:
        """
        Update parameter estimates based on a new observation.

        Parameters
        ----------
        prices : List[float]
            The prices used in the new observation
        choices : List[int]
            The choices made by customers in the new observation

        Updates
        -------
        self.system : TieredPricingSystem
            Updated system with new parameter estimates
        self.a_mean : float
            Updated mean of lower bound parameter
        self.b_mean : float
            Updated mean of upper bound parameter
        self.a_posterior : List[float]
            Updated posterior samples of lower bound parameter
        self.b_posterior : List[float]
            Updated posterior samples of upper bound parameter
        self.lambda_posterior : List[float]
            Updated posterior samples of lambda parameter
        self.lambda_mean : float
            Updated mean of lambda parameter
        self.likelihood_posterior : List[float]
            Updated log likelihoods
        self.prev_trials : List[Trial]
            Updated list of previous trials
        """
        curr_trial = Trial(prices, choices)
        self.prev_trials.append(curr_trial)

        new_weights = np.zeros(self.num_samples)
        for i, (a, b, lam) in enumerate(self.particles):
            prob = self.param_probability(a, b, lam, curr_trial)
            if prob > 0:
                new_weights[i] = self.weights[i] * (prob**0.5)
            else:
                new_weights[i] = 0

        sum_weights = np.sum(new_weights)
        self.weights = new_weights / sum_weights
        self.a_mean = np.sum(self.particles[:, 0] * self.weights)
        self.b_mean = np.sum(self.particles[:, 1] * self.weights)
        self.lambda_mean = np.sum(self.particles[:, 2] * self.weights)

        self.system.update_parameters(
            (self.a_mean + self.b_mean) / 2,
            (self.b_mean - self.a_mean) / 2,
            self.lambda_mean,
        )
