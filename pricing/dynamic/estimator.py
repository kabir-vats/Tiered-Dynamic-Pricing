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
        a_prior: Tuple[float, float] = (0.1, 10),
        b_prior: Tuple[float, float] = (10, 10),
        lam_prior: Tuple[float, float] = (0.5, 0.4),
        num_samples: int = 1000,
    ):
        self.a_mean, self.a_std = a_prior
        self.b_mean, self.b_std = b_prior
        self.lambda_mean, self.lambda_std = lam_prior
        self.num_samples = num_samples
        self.system = system
        self.prev_trials = []

        self.a_posterior = []
        self.b_posterior = []
        self.lambda_posterior = []
        self.likelihood_posterior = []  # Initialize with log(1) = 0

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
        # Update previous probabilities based on new data
        for i in range(len(self.likelihood_posterior)):
            if np.isfinite(self.likelihood_posterior[i]):
                param_prob = self.param_probability(
                    self.a_posterior[i],
                    self.b_posterior[i],
                    self.lambda_posterior[i],
                    curr_trial,
                )
                if param_prob > 0:
                    self.likelihood_posterior[i] += math.log(param_prob)
                else:
                    self.likelihood_posterior[i] = -np.inf

        # Sample parameter values from prior distributions
        a_samples = np.random.uniform(
            self.a_mean - self.a_std, self.a_mean + self.a_std, self.num_samples
        )
        b_samples = np.random.uniform(
            self.b_mean - self.b_std, self.a_mean + self.b_std, self.num_samples
        )
        lambda_samples = np.random.uniform(
            self.lambda_mean - self.lambda_std,
            self.lambda_mean + self.lambda_std,
            self.num_samples,
        )

        # Ensure valid values
        lambda_samples = lambda_samples[(lambda_samples >= 0) & (lambda_samples <= 1)]
        a_sam = []
        b_sam = []
        lam_sam = []
        for a, b, lam in zip(a_samples, b_samples, lambda_samples):
            if b > a:
                a_sam.append(a)
                b_sam.append(b)
                lam_sam.append(lam)

        # Compute likelihoods for each sample
        log_likelihoods = []
        valid_choices = 0

        for a, b, lam in zip(a_sam, b_sam, lam_sam):
            param_prob = self.param_probability(a, b, lam, curr_trial)
            if param_prob > 0:
                log_prob = math.log(param_prob)
            else:
                log_prob = -np.inf
            for trial in self.prev_trials:
                if not np.isfinite(log_prob):
                    break
                param_prob = self.param_probability(a, b, lam, trial)
                if param_prob > 0:
                    log_prob += math.log(param_prob)
                else:
                    log_prob = -np.inf
            if np.isfinite(log_prob):
                valid_choices += 1
            log_likelihoods.append(log_prob)

        if valid_choices == 0:
            self.a_std *= 1.1
            self.b_std *= 1.1
            self.system.update_parameters(
                (self.a_mean + self.b_mean) / 2,
                (self.b_mean - self.a_mean) / 2,
                self.lambda_mean,
            )
            return

        log_likelihoods = np.array(log_likelihoods)
        max_log = np.max(log_likelihoods)
        log_likelihoods = log_likelihoods - max_log

        likelihoods = np.exp(log_likelihoods)

        posterior_weights = likelihoods / np.sum(likelihoods)

        self.prev_trials.append(curr_trial)

        self.likelihood_posterior.append(
            math.log(np.sum(likelihoods) / valid_choices) + max_log
        )
        self.a_posterior.append(np.sum(posterior_weights * a_sam))
        self.b_posterior.append(np.sum(posterior_weights * b_sam))
        self.lambda_posterior.append(np.sum(posterior_weights * lam_sam))

        recent_weights = np.exp(
            self.likelihood_posterior - np.max(self.likelihood_posterior)
        )
        recent_weights = recent_weights / np.sum(recent_weights)

        self.a_mean = np.sum(np.array(self.a_posterior) * recent_weights)
        self.b_mean = np.sum(np.array(self.b_posterior) * recent_weights)
        self.lambda_mean = np.sum(np.array(self.lambda_posterior) * recent_weights)

        self.system.update_parameters(
            (self.a_mean + self.b_mean) / 2,
            (self.b_mean - self.a_mean) / 2,
            self.lambda_mean,
        )


class ParticleBayesianEstimator:
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
        a_prior: Tuple[float, float] = (-10, 10),
        b_prior: Tuple[float, float] = (0, 20),
        lam_prior: Tuple[float, float] = (0, 1),
        num_samples: int = 10000,
    ):
        self.particles = np.zeros((num_samples, 3))
        self.particles[:, 0] = np.random.uniform(
            a_prior[0], a_prior[1], num_samples
        )
        self.particles[:, 1] = np.random.uniform(
            b_prior[0], b_prior[1], num_samples
        )
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
                new_weights[i] = self.weights[i] * prob
            else:
                new_weights[i] = 0

        sum_weights = np.sum(new_weights)
        self.weights = new_weights / sum_weights
        self.a_mean = np.sum(self.particles[:, 0] * self.weights)
        self.b_mean = np.sum(self.particles[:, 1] * self.weights)
        self.lambda_mean = np.sum(self.particles[:, 2] * self.weights)

        self.system.update_parameters((self.a_mean + self.b_mean) / 2,
                                        (self.b_mean - self.a_mean) / 2,
                                        self.lambda_mean)