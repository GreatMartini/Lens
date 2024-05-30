################################### Reconstructs parameters of the lens via the metropolis hastings algorithm ######################

import matplotlib.pyplot as plt
import numpy as np
from .statfunc import *

def Metropolis():
    f_0, sigma_0 = 0.5, 0.5                                                                             # Initial choises for the parameters                                                 
    theta_init = np.array([sigma_0, f_0])

    x1 = np.linspace(-40, 40, 2000)
    x2 = np.linspace(-40, 40, 2000)
    dx1, dx2 = x1[1]- x1[0], x2[1]-x2[0]                                                                # Regular grid
    X1, X2 = np.meshgrid(x1, x2)
    like = 5.0
    k = 0
    while(np.abs(like) >= 0.1):
        if (k<=1000):
            theta = theta_prime(theta_init)
            pos = lens_solve(theta_init, X1, X2, dx1, dx2, xc)
            posp = lens_solve(theta, X1, X2, dx1, dx2, xc)
            Chi2 = cost_function(images,pos)
            Chi2p = cost_function(images,posp)
            prior = log_prior(theta_init) #Meter condicion de infinito
            priorp = log_prior(theta) # Meter condicion de infinito
            like = log_likelihood(Chi2)
            likep = log_likelihood(Chi2p)
            post = log_posterior(like, prior)
            postp = log_posterior(likep, priorp)
            alpha = acceptance(postp, post)
            theta_init = test(alpha, theta, theta_init)
            print(theta_init)
        else:
            break
        k += 1
    params = theta_init