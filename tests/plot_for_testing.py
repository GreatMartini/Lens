import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.integrate import solve_ivp
import sys
from scipy.fft import fftn, ifftn
from pde import PDE, FieldCollection, CartesianGrid, ScalarField
"""
lens = h5py.File("/Users/bach/Desktop/Lensing/Lensing/output/Lens_test.hdf5", 'r')
image = lens["Image"]

xi_0 = lens["System Parameters"].attrs["Xi0"]
eta_0 = lens["System Parameters"].attrs["Eta0"]
y1 = lens["Source Parameters"].attrs["x0"]/eta_0
y2 = lens["Source Parameters"].attrs["y0"]/eta_0

Ry = np.sqrt(y1**2+y2**2)
x1 = lens["Coordinates"]["x1"]/xi_0
x2 = lens["Coordinates"]["x2"]/xi_0
xc = lens["Lens Parameters"].attrs["xc"]/lens["System Parameters"].attrs["Xi0"]
rayon_einstein = np.sqrt(1-2*xc)
print(rayon_einstein)
plt.imshow(image, cmap = "gray", extent = [np.min(x1), np.max(x1), np.min(x2), np.max(x2)])
plt.show()
plt.close()
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
eta_0 = D_l/(D_s*xi_0)                                                      # Scaling parameter
alpha_scale = D_ls*D_l/(D_s*xi_0)                                           # The scaling factor that is applied to alpha when the equation is adimensionated
dens_critic = c**2/(4*np.pi*G)*D_s/(D_l*D_ls)

class source:                                                               # Creates a source
    """  Class representing a source
    
    Parameters:
    ------------

    extended = False : bool
        If True the source is taken to be extended if False the source is point-like  
        (At this stage of the development only point-like sources are allowed).
    r : float
        Radius of the extended source.
    N : float
        Number of points of the extended source

    Attributes:
    ------------
    
    x : float, 1darray
        Coordinates of the source on the semi-major axis, float if point-like and array if extended source.
    y : float, 1darray
        Coordinates of the source on the semi-minor axis, float if point-like and array if extended source.
    z : float
        Coordinate of the source plane on the orthogonal axis to the source and lens planes.
    
    Methods:
    --------

    construct(r, N) : 
        Constructs an extended source with the specified radius and number of points.
        
    """
    
    def construct(self, r, N):                                              # Builds the source for non punctual sources, r being the radius and N the total number of points
        """     
        Parameters:
        ------------
        
        r : float
            Radius of the extended source.
        N : float
            Number of points of the extended source
        
        """                                 
        Nr = int(np.sqrt(r*N))                                              # Number of points in r (defined by scale factor of the transformation)                                              
        Ntheta = int(N/Nr)                                                  # Number of points in theta (angle)
        theta = np.linspace(0.0001, 2*np.pi, Ntheta)                        # Create the angle vector
        rv = np.linspace(0, r, Nr)                                          # Create the radius vector
        x = rv[None, :]*np.cos(theta[:, None])                              # Build x
        y = rv[None, :]*np.sin(theta)[:, None]                              # Build y
        x = np.reshape(x, (Ntheta*rv.size))                                 # Reshape
        y = np.reshape(y, (Ntheta*rv.size))
        return x, y                                                         # Return cartesian coordinates        

    def __init__(self, extended = False, r = 0, N = 1):                     # Initialise source
        if extended == False:                                               # Point source
            self.x = xs0
            self.y = ys0
            self.z = zs0
        if extended == True:                                                # Extended source centered around initial coordinates
            self.x = self.construct(r, N)[0] + x0
            self.y = self.construct(r, N)[1] + y0
            self.z = z0


class lens:                                                                 # Creates a lens
    """  Class representing a the mass distribution of a gravitational lens    

    Attributes:
    ------------
    
    x0 : float, 1darray
        Coordinates of the source on the semi-major axis, float if point-like and array if extended source.
    y0 : float, 1darray
        Coordinates of the source on the semi-minor axis, float if point-like and array if extended source.
    z0 : float
        Coordinate of the source plane on the orthogonal axis to the source and lens planes.
    x1 : 2darray
        Contains the coordinates of the semi-major axis over the grid.
    x2 : 2darray
        Contains the coordinates of the semi-minor axis over the grid.
    density_surf : 2darray
        Contains the grid values of the reduced isothermal surface density. (Only isothermal density profiles are allowed at this stage of the development)
    dx1 : float
        Spatial step in the direction of the semi-major axis.
    dx2 : float
        Spatial step in the direction of the semi-minor axis.
    
    Methods:
    --------

    build_mesh() : 
        Creates the coordinate meshgrid.
    NIE():
        Creates a non-singular isothermal elliptic surface density.

    """
    def build_mesh(self):                                                   # Builds the mesh grid for the lens
        """     
        Returns:
        ------------

        X1, X2 : 2darrays
            Meshgrid containing the coordinates of the axes.
    
        """ 
        x1_dom = np.linspace(x1_l, x1_u, Nx1)                               # We choose first to create the domain, then calculate the dsteps
        x2_dom = np.linspace(x2_l, x2_u, Nx2)                               # and finaly to create the ghost zone, as this method provides
        dx1 = x1_dom[1] - x1_dom[0]                                         # that the simulation passes through the bounds and dsteps
        dx2 = x2_dom[1] - x2_dom[0]                                         # will be returned precisely
        x1_ghost = np.linspace(x1_u + dx1, x1_u + N_ghost*dx1, N_ghost)     # Pseudo-ghost points in X, on each side
        x2_ghost = np.linspace(x2_u + dx2, x2_u + N_ghost*dx2, N_ghost)     # Pseudo-ghost points in Y, on each side
        x1_left = np.concatenate((np.flip(-x1_ghost), x1_dom))              # Creation of the total domain  
        x1 = np.concatenate((x1_left, x1_ghost))
        x2_left = np.concatenate((np.flip(-x2_ghost), x2_dom))
        x2 = np.concatenate((x2_left, x2_ghost))
        X1, X2 = np.meshgrid(x1, x2)
        return X1, X2                                                       # Returns the meshgrid                                                                                    
    def NIE(self, x1, x2):                                                  # General non-singular isothermal lens, reduced surface density
        """
        Parameters:
        -----------

        X1, X2 : 2darrays
            Meshgrid containing the coordinates of the axes.

        Returns:
        --------
        
            dens : Non-singular isothermal reduced surface density. 

        """ 
        return (np.sqrt(f))/(2*np.sqrt(x1**2 + (f*x2)**2 + xc**2))               
    def __init__(self):                                                     # Builds the lens
        self.x0 = xl0
        self.y0 = yl0
        self.z0 = zl0
        self.x1 = self.build_mesh()[0]
        self.x2 = self.build_mesh()[1]
        self.density_surf = self.NIE(self.x1, self.x2)
        self.dx1 = self.x1[0, 1] - self.x1[0, 0]
        self.dx2 = self.x2[1, 0] - self.x2[0, 0]

def poisson(field, x1, x2, dx1, dx2):                                       # Solves the Poisson equation using Fourier Transform and zero padding
    """  Solves the poissson equation via the fourier transform 
    
    Parameters:
    ------------
    
    field : 2darray
        Scalar field over which we want to solve the poisson equation.
    dx1 : float
        Step size along the semi-major axis.
    dx2 : float
        Step size along the semi-minor axis.
    
    Returns:
    --------
    f : 2darray
        Scalar solution of the poisson equation over a grid of the same size as the field argument.
    """
    N1 = np.shape(field)[0] 
    N2 = np.shape(field)[0]
    Ntot1 = N1 #+ 2*N_ghost                                                 # We define the total number of points of the array                                                 
    Ntot2 = N2 #+ 2*N_ghost
    Npad1 = Ntot1                                                           # We define the size of the padding for each side of axis 1, columns
    Npad2 = Ntot2                                                           # We define the size of the padding for each side of axis 2, rows
    fantom_1 = np.zeros((Ntot2, Npad1))                                     # We define the padding along axis 1: columns (On one side)
    fantom_2 = np.zeros((Npad2, Ntot1+2*Npad1))                             # We define the padding along axis 2: Rows (On one side) taking into account the padding on axis 1
    field_1 = np.concatenate((fantom_1, field), axis = 1)                   # We pad along x, left
    field_2 = np.concatenate((field_1, fantom_1), axis = 1)                 # We pad along x, right                  
    field_3 = np.concatenate((fantom_2, field_2))                           # We pad along y, top
    field_4 = np.concatenate((field_3, fantom_2))                           # we pad along y, bottom
    Nk2, Nk1 = np.shape(field_4)                                            # Define the vectors length in the Fourier space
    kx = 2*np.pi*np.fft.fftfreq(Nk1, d = dx1)                               # Compute the Fourier vector's length
    ky = 2*np.pi*np.fft.fftfreq(Nk2, d = dx2)

    Kx, Ky = np.meshgrid(kx,ky)                                             # Create mesh
    K_squared = Kx**2 + Ky**2                                               # Computes the norm of the Fourier vectors
    K_squared[0,0] = 1                                                      # Avoid singularities
    g = fftn(field_4)                                                       # Fourier transform of the field
    ft = 2*g/(K_squared)                                                        # Compute the laplacian in the fourier space
    f = -np.real(ifftn(ft))[Npad2:-Npad2,Npad1:-Npad1]                       # Returns the solution of the poisson equation, slices the padding
    return f



def numerical_poisson(r, y, x):
    psi = y[0]
    sig = y[1]
    dpsi_dr = sig 
    if r == 0:
        dsig_dr = 0
    else: 
        dsig_dr = 1/np.sqrt(r**2 + x**2) -(1/r) * sig 
    return [dpsi_dr, dsig_dr]


epsilon = 2
lens1 = lens()
surf_dens = lens1.density_surf
X1, X2 = lens1.x1, lens1.x2 
dx1, dx2 = X1[0,1]-X1[0,0], X2[1,0]-X2[0,0]
psi = poisson(surf_dens, X1, X2, dx1, dx2)

