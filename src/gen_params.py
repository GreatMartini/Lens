""" Parameter file for the initial image generation from an isothermal projected mass distribution

Source Parameters:
------------------

zs0 : float
    Coordinate of the source plane on the axis orthogonal to both the source's and the lens' plane.    
xs0 : float
    Coordinate of the point-like source in the semi-major axis.
ys0 : float
    Coordinate of the point-like source in the semi-minor axis.

Lens Parameters:
------------------

Sigma : float 
    Dispersion velocity for the isothermal mass distribution.
f : float
    Ellipticity of the isothermal mass distribution, must be between (0,1].
Nx1, Nx2 : int
    Number of grid points spanning the lens' plane in the semi-major and semi-minor axis respectively, at this stage of the developpement must be odd 
    and equal. The total number of grid points is Nx1*Nx2.
x1_l, x1_u : float
    Lower and upper boundaries for the semi-major axis respectively.
x2_l, x2_u : float
    Lower and upper boundaries for the semi-minor axis respectively.
N_ghost : int
    Number of ghost points on each side of the two axis. This allows to compute the finite differences derivatives without the use of boundary conditions (for the moment).
    At each derivative the size of resulting array will be reduced by 2 (1 point on each side) on each axis hence, the resulting size will be (N1-2)*(N2-2).
G : float
    Universal constant of gravitation. In natural units = 1.
c : float
    Speed of light in vacuum. In natural units = 1.
xl0 : float
    Semi-major axis coordinate of the lens plane center.
yl0 : float
    Semi-minor axis coordinate of the lens plane center.
zl0 : float
    Coordinate of the lens plane center on the axis orthogonal to the source's and lens' planes.

Scale parameters:
-----------------

D_s : float
    Distance from the observer to the source plane.                                                            
D_l : float
    Distance from the observer to the lens plane.                                     
D_ls : float
    Distance between the source and the lens planes.
xi_0 : float
    Scale parameter for the coordinates on the lens plane.
eta_0 : float
    Scale parameter for the coordinates on the source plane.
alpha_scale : float
    Scale paremeter for the angle of deviation.                                
dens_critic : float
    Critical density for the scaling of the 2-dimensional mass distribution.
"""
import numpy as np

# Source parameters                                 
zs0 =  3.                                                                   # Position of the source's center (x and y in reduced units)
xs0 = 0 
ys0 = 0

# Lens parameters
Sigma = 2.5                                                                   #(1-np.random.rand())*5       # Cannot be 0        
f = 1                                                                       #1-np.random.rand()           # Cannot be 0
# Lens' mesh parameters
Nx1,  Nx2 = 1001, 1001                                                      # Number of points for the lens' domain in the principal and secondary axis, Array better be ood
x1_l, x1_u = -2., 2.                                                        # Bounds for the principal axis of the lens domain          
x2_l, x2_u = -2., 2.                                                        # Bounds for the secondary axis of the lens domain
N_ghost = 4                                                                 # Number of ghost points N under and N above the limits of the array for each axis
                                                                            # Allows to compute N_ghost derivatives over the domain without providing boundary conditions 
                                                                            # Better be chosen in order to retrieve an odd array       
                                                                            # The code is created for squared arrays
G, c = 1, 1                                                                 # Gravitational Constant, Speed of light: natural units
xl0, yl0, zl0 = 0., 0., 1.                                                  # Position of the lens' center
xc = 0.05                                                                    # Critical radius for softened spherical lenses
# Distance parameters for the "reduction" of the lens equation 
D_s = zs0                                                                   # Distance to the source
D_l = zl0                                                                   # Distance to the lens                                          
D_ls = D_s-D_l                                                              # Distance between the lens and the source
xi_0 = 4*np.pi*((Sigma/c)**2)*D_ls/D_s*D_l                                  # Scaling parameter: If we change it we will have to change the form of the isothermal potential
eta_0 = xi_0*D_s/D_l#D_l/(D_s*xi_0)                                                      # Scaling parameter
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
