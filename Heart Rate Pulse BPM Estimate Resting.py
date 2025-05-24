# -*- coding: utf-8 -*-
"""
Created on Thu May 22 20:35:50 2025

@author: Joben Gersalia
"""

# A cardiologist wants to estimate the average resting heart rate (beats per minute) in a hospital in Pasay.
# Prior knowledge suggests it’s about 72 bpm, but the actual value may differ. 
# A new sample of 75 patients is gathered to update this belief using Bayesian inference.

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data (observed heart rates from the sample)
np.random.seed(2)  # For reproducibility
true_mean_hr = 70  # True population mean heart rate (unknown to researcher)
true_hr_sd = 8     # True population standard deviation of heart rate
sample_size = 75
heart_rates = np.random.normal(true_mean_hr, true_hr_sd, size=sample_size)

# Step 2: Define the prior distribution for mean and variance
prior_mu_mean = 72         # Prior belief: average resting heart rate is around 72 bpm
prior_mu_precision = 0.02  # Precision of prior mean (low precision => high uncertainty)
prior_sigma_alpha = 2      # Prior shape parameter for sigma² (weakly informative)
prior_sigma_beta = 30      # Prior scale parameter for sigma²

# Step 3: Update posterior parameters based on observed data
posterior_mu_precision = prior_mu_precision + sample_size / true_hr_sd**2
posterior_mu_mean = (
    prior_mu_precision * prior_mu_mean + np.sum(heart_rates) / true_hr_sd**2
) / posterior_mu_precision
posterior_sigma_alpha = prior_sigma_alpha + sample_size / 2
posterior_sigma_beta = prior_sigma_beta + 0.5 * np.sum((heart_rates - np.mean(heart_rates))**2)

# Step 4: Draw samples from the posterior distributions
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.sqrt(1 / np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

# Step 5: Visualize posterior distributions
plt.figure(figsize=(10, 4))

# Histogram of posterior samples for mean heart rate
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, color='skyblue', edgecolor='black', density=True)
plt.title('Posterior of $\mu$ (Heart Rate)')
plt.xlabel('$\mu$ (bpm)')
plt.ylabel('Density')

# Histogram of posterior samples for standard deviation
plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, color='lightgreen', edgecolor='black', density=True)
plt.title('Posterior of $\sigma$ (Heart Rate)')
plt.xlabel('$\sigma$ (bpm)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Step 6: Report posterior summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu (Heart Rate):", mean_mu)
print("Standard deviation of mu (Heart Rate):", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma (Heart Rate):", mean_sigma)
print("Standard deviation of sigma (Heart Rate):", std_sigma)
