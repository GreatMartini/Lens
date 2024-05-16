################################################## Code parameters ##############################
import numpy as np
# Source parameters                                 
zs0 =  3.                                                                   # Position of the source's center (x and y in reduced units)
xs0 = 0 
ys0 = 0

# Lens parameters
Sigma = np.random.rand()+0.01    # Cannot be 0        
f = np.random.rand()+0.001           # Cannot be 0
# Lens' mesh parameters
Nx1,  Nx2 = 1001, 1001                                                      # Number of points for the lens' domain in the principal and secondary axis, Array better be ood
x1_l, x1_u = -2., 2.                                                        # Bounds for the principal axis of the lens domain          
x2_l, x2_u = -2., 2.                                                        # Bounds for the secondary axis of the lens domain
N_ghost = 4                                                                 # Number of ghost points N under and N above the limits of the array for each axis
                                                                            # Allows to compute N_ghost derivatives over the domain without providing boundary conditions 
                                                                            # Better be chosen in order to retrieve an odd array       
                                                                            # The code is created for squared arrays
# Lens' potential parameters
G, c = 1, 1                                                                 # Gravitational Constant, Speed of light: natural units
xl0, yl0, zl0 = 0., 0., 1.                                                  # Position of the lens' center
xc = 0.05                                                                    # Critical radius for softened spherical lenses
# Distance parameters for the "reduction" of the lens equation 
D_s = zs0                                                                   # Distance to the source
D_l = zl0                                                                   # Distance to the lens                                          
D_ls = D_s-D_l                                                              # Distance between the lens and the source
xi_0 = 4*np.pi*((Sigma/c)**2)*D_ls/D_s*D_l                                  # Scaling parameter: If we change it we will have to change the form of the isothermal potential
eta_0 = D_l/(D_s*xi_0)                                                      # Scaling parameter
alpha_scale = D_ls*D_l/(D_s*xi_0)                                           # The scaling factor that is applied to alpha when the equation is adimensionated
dens_critic = c**2/(4*np.pi*G)*D_s/(D_l*D_ls)

                                                                            # It is good to have it in order to recover the real deflection angle
# The physical lens equation is:
# D_s*beta = D_s*theta-alpha_D_ls
# eta = D_s*beta
# xi = D_l*theta
# xi_0 = arbitrary
# eta_0 = xi_0*D_l/D_s
# alpha = alpha_scale*alpha_scaled
# y = eta/eta_0
# x = xi/xi_0
# And we have the scaled and adimensioned lens equation:
# y = x-alpha_scaled 
