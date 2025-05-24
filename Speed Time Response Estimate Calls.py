# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:01:54 2025

@author: Joben Gersalia
"""
#A company wants to estimate the average customer service response time. They guess it's around 2 minutes, but aren't sure. They collect 120 call logs and plan use Bayesian inference to update their estimate of the average time and how much it varies.

import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate response time data
np.random.seed(46)
true_mean_resp = 2.2
true_sd_resp = 0.5
sample_size = 120
response_times = np.random.normal(true_mean_resp, true_sd_resp, size=sample_size)

# Step 2: Prior beliefs
prior_mu_mean = 2.0
prior_mu_precision = 0.1
prior_sigma_alpha = 3
prior_sigma_beta = 1

# Step 3: Update
posterior_mu_precision = prior_mu_precision + sample_size / true_sd_resp**2
posterior_mu_mean = (
    prior_mu_precision * prior_mu_mean + np.sum(response_times) / true_sd_resp**2
) / posterior_mu_precision

posterior_sigma_alpha = prior_sigma_alpha + sample_size / 2
posterior_sigma_beta = prior_sigma_beta + 0.5 * np.sum((response_times - np.mean(response_times))**2)

# Step 4: Draw samples
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_sigma = np.sqrt(1 / np.random.gamma(posterior_sigma_alpha, 1 / posterior_sigma_beta, size=10000))

# Step 5: Plotting
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='plum', edgecolor='black')
plt.title('Posterior of $\mu$ (Response Time)')
plt.xlabel('$\mu$ (minutes)')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='mediumseagreen', edgecolor='black')
plt.title('Posterior of $\sigma$ (Response Time)')
plt.xlabel('$\sigma$ (minutes)')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# Step 6: Summary
print("Mean of μ (Response Time):", np.mean(posterior_mu))
print("SD of μ (Response Time):", np.std(posterior_mu))
print("Mean of σ (Response Time):", np.mean(posterior_sigma))
print("SD of σ (Response Time):", np.std(posterior_sigma))
