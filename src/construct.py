""" Module containing the classes. The classes are one for the source (point-like only at this stage of development) and for the lens
(isothermal only at this stage of development)."""
from .gen_params import *

##################################### Classes: Lens and Sources #####################################
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