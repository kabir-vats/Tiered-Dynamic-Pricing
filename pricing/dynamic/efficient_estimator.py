from typing import Tuple, List
import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from scipy.special import logsumexp
from collections import Counter


class Trial:
    def __init__(self, prices: List[float], choices: List[int]):
        self.prices = prices
        self.choices = choices
        self.counts = Counter(choices)


class EfficientGaussianEstimator:
    def __init__(
        self,
        system,
        mu_prior: Tuple[float, float] = (0, 5),
        sigma_prior: Tuple[float, float] = (0, 5),
        lam_prior: Tuple[float, float] = (0, 1),
        num_samples: int = 1024,
    ):
        self.system = system
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.lam_prior = lam_prior
        self.num_samples = num_samples

        self.particles = self._generate_particles()
        self.weights = np.ones(self.num_samples) / self.num_samples
        self.prev_trials = []

    def _generate_particles(self):
        sampler = Sobol(d=3, scramble=True)
        raw = sampler.random_base2(m=int(np.log2(self.num_samples)))
        mus = self.mu_prior[0] + raw[:, 0] * (self.mu_prior[1] - self.mu_prior[0])
        sigmas = self.sigma_prior[0] + raw[:, 1] * (self.sigma_prior[1] - self.sigma_prior[0])
        lams = self.lam_prior[0] + raw[:, 2] * (self.lam_prior[1] - self.lam_prior[0])
        return np.stack((mus, sigmas, lams), axis=1)

    def _param_log_probs_batch(self, trial: Trial) -> np.ndarray:
        mus = self.particles[:, 0]
        sigmas = self.particles[:, 1]
        lams = self.particles[:, 2]

        original_params = (self.system.mu, self.system.sigma, self.system.lam)
        base_intervals = np.zeros((len(self.system.costs)+1, 2, self.num_samples))

        for i, (mu, sigma, lam) in enumerate(self.particles):
            if self.weights[i] == 0:
                continue
            self.system.update_parameters(mu, sigma, lam)
            base_intervals[:, :, i] = self.system.calculate_intervals(trial.prices)

        self.system.update_parameters(*original_params)

        log_probs = np.zeros(self.num_samples)
        for tier, count in trial.counts.items():
            if count == 0:
                continue
            lower, upper = base_intervals[tier]
            z_upper = (upper - mus) / sigmas
            z_lower = (lower - mus) / sigmas
            tier_probs = np.clip(norm.cdf(z_upper) - norm.cdf(z_lower), 1e-10, 1.0)
            log_probs += count * np.log(tier_probs)

        return log_probs

    def _resample_particles(self):
        # print('test')
        indices = np.random.choice(len(self.particles), size=self.num_samples, p=self.weights)
        resampled_particles = self.particles[indices]
        noise = np.random.normal(loc=0.0, scale=(0.05, 0.05, 0.01), size=resampled_particles.shape)
        jittered = resampled_particles + noise
        self.particles = jittered
        self.weights = np.ones(self.num_samples) / self.num_samples

    def update(self, prices: List[float], choices: List[int]):
        trial = Trial(prices, choices)
        self.prev_trials.append(trial)

        log_probs = self._param_log_probs_batch(trial)
        log_weights = np.log(self.weights + 1e-12) + 0.5 * log_probs
        log_weights -= logsumexp(log_weights)
        self.weights = np.exp(log_weights)

        if np.sum(self.weights > 1e-4) < self.num_samples * 0.5:
            self._resample_particles()

        self.mu_mean = np.sum(self.particles[:, 0] * self.weights)
        self.sigma_mean = np.sum(self.particles[:, 1] * self.weights)
        self.lambda_mean = np.sum(self.particles[:, 2] * self.weights)

        self.system.update_parameters(
            self.mu_mean, self.sigma_mean, self.lambda_mean
        )

        return self.mu_mean, self.sigma_mean, self.lambda_mean
