# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:21:21 2025

@author: Joben Gersalia
"""
#A transportation researcher wants to figure out how long workers in Manila City usually spend commuting. Previous studies suggest it's about 30 minutes, but there’s still some uncertainty. To get a clearer picture, they surveyed 100 people and plan to use Bayesian inference to update their estimate based on this new data.

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data (observed commute times)
np.random.seed(42)
true_mean_commute = 32  # Actual average commute time
true_commute_sd = 10    # Standard deviation in commute time
sample_size = 100
commute_times = np.random.normal(true_mean_commute, true_commute_sd, size=sample_size)

# Step 2: Define the prior
prior_mu_mean = 30         # Prior belief about the mean commute time
prior_mu_precision = 0.01  # Low precision => high uncertainty
prior_sigma_alpha = 2      # Weak prior on standard deviation
prior_sigma_beta = 30      

# Step 3: Update posterior parameters
posterior_mu_precision = prior_mu_precision + sample_size / true_commute_sd**2
posterior_mu_mean = (
    prior_mu_precision * prior_mu_mean + np.sum(commute_times) / true_commute_sd**2
) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + sample_size / 2
posterior_sigma_beta = prior_sigma_beta + 0.5 * np.sum((commute_times - np.mean(commute_times))**2)

# Step 4: Sample from the posterior
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.sqrt(1 / np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

# Step 5: Plot the posterior distributions
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.title('Posterior distribution of $\mu$ (Commute Time)')
plt.xlabel('$\mu$ (minutes)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.title('Posterior distribution of $\sigma$ (Commute Time)')
plt.xlabel('$\sigma$ (minutes)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Step 6: Report summary statistics
mean_mu = np.mean(posterior_mu)
std_mu = np.std(posterior_mu)
print("Mean of mu (Commute Time):", mean_mu)
print("Standard deviation of mu (Commute Time):", std_mu)

mean_sigma = np.mean(posterior_sigma)
std_sigma = np.std(posterior_sigma)
print("Mean of sigma (Commute Time):", mean_sigma)
print("Standard deviation of sigma (Commute Time):", std_sigma)
