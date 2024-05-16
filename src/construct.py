##################################### Script that creates classes ##################################
from gen_params import *

##################################### Classes: Lens and Sources #####################################
class source:                                                               # Creates a source
    def construct(self, r, N):                                              # Builds the source for non punctual sources, r being the radius and N the total number of points                                 
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
    def build_mesh(self):                                                   # Builds the mesh grid for the lens
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
        return (np.sqrt(f))/(np.sqrt(x1**2 + (f*x2)**2 + xc**2))               
    def __init__(self):                                                     # Builds the lens
        self.x0 = xl0
        self.y0 = yl0
        self.z0 = zl0
        self.x1 = self.build_mesh()[0]
        self.x2 = self.build_mesh()[1]
        self.density_surf = self.NIE(self.x1, self.x2)
        self.dx1 = self.x1[0, 1] - self.x1[0, 0]
        self.dx2 = self.x2[1, 0] - self.x2[0, 0]       