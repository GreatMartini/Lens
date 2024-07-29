""" Module that reconstructs the parameters (velocity dispersion, ellipticity)of the isothermal gravitational potential
via the metropolis-hastings algorithm."""
import matplotlib.pyplot as plt
import numpy as np
from .statfunc import *

def Metropolis():
    """
    Function that implements the metropolis-hastings algorithm to reconstruct the ellipticity and velocity dispersion
    of the isothermal gravitational potential.
    
    Returns:
    --------
    params : 1darray
        Containing the estimated values of the dispersion velocity and ellipticity.
    """
    f_0, sigma_0 = 0.5, 0.5                                     # Initial choises for the parameters                                                 
    theta_init = np.array([sigma_0, f_0])                       # Array storing the initial choises

    x1 = np.linspace(-40, 40, 2000)                             # New coordinates
    x2 = np.linspace(-40, 40, 2000)                             # New coordinates             
    dx1, dx2 = x1[1]- x1[0], x2[1]-x2[0]                        # Regular grid
    X1, X2 = np.meshgrid(x1, x2)                                # Meshgrid
    like = 5.0                                                  # Initial value for the logarithm of the likelihood (can be anything of big abs value)
    k = 0                                                       # Iteration parameter for stopping the algorithm
    while(np.abs(like) >= 0.1):
        if (k<=1000):
            theta = theta_prime(theta_init)                         # Computes the values of the second set of parameters
            pos = lens_solve(theta_init, X1, X2, dx1, dx2, xc)      # Calculates the images' positions given the first set of parameters
            posp = lens_solve(theta, X1, X2, dx1, dx2, xc)          # Calculates the images's positions given the second set of parameters
            Chi2 = cost_function(images,pos)                        # Calculates the cost function of the first set of parameters
            Chi2p = cost_function(images,posp)                      # Calculates the cost function of the second set of parameters
            prior = log_prior(theta_init)                           # Implements the prior of the first set of parameters 
            priorp = log_prior(theta)                               # Implements the prior of the second set of parameters 
            like = log_likelihood(Chi2)                             # Calculates the logarithm of the likelihood for the first set of parameters
            likep = log_likelihood(Chi2p)                           # Calculates the logarithm of the likelihood for the second set of parameters 
            post = log_posterior(like, prior)                       # Calculates the posterior for the first set of parameters
            postp = log_posterior(likep, priorp)                    # Calculates the posterior for the second set of parameters
            alpha = acceptance(postp, post)                         # Calculates the acceptance
            theta_init = alpha_test(alpha, theta, theta_init)       # Tests the likelihood ratio and select the most likely parameters
            print(theta_init)
        else:
            break
        k += 1
    params = theta_init                                             # Return the estimated parameters
    return params