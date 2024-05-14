import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
import h5py

####################################### Important Information ##########################################
# In this code the Number of points along each axis in the lens' mesh must be equal (and the spatial steps), 
# The code can be easily adapted in order to do rectangular meshgrids
# In order to do so, the code must be modified in the poisson, grad, and divergence functions
# The poisson function supposes that the size of the array is the original one so it must be modified
# If we want to use it in an other context because, in this code the derivatives reduce the array of 2 
# At each instance
# Sometimes the code will crash because the parameters will cause the root finding function to diverge
# We only would need to generate the images with other random parameters
########################################################################################################
# Despues de reconstruir verificar dimensionalización
###################################### Parameters ######################################################
def Generate(xs0, ys0, Sigma, f):
    # Source parameters                                 
    zs0 =  3.                                                                   # Position of the source's center (x and y in reduced units)

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
        
    ############################################# Functions #############################################
    def poisson(field, x1, x2, dx1, dx2):                                       # Solves the Poisson equation using Fourier Transform and zero padding
        Ntot1 = Nx1 + 2*N_ghost                                                 # We define the total number of points of the array                                                 
        Ntot2 = Nx2 + 2*N_ghost
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
        ft = g/K_squared                                                        # Compute the laplacian in the fourier space
        f = np.real(ifftn(ft))[Npad2:-Npad2,Npad1:-Npad1]                       # Returns the solution of the poisson equation, slices the padding
        return -f
    def grad(f, dx , dy):                                                       # Computes the gradient using centered finite differences and ghost points: Returns two arrays (N1-2, N2-2)
        N = np.shape(f)[0]                                                      # Extract the shape of the mesh (squared) 
        sparseA = -np.eye(N,N,-1)+np.eye(N,N,1)                                 # Create first sparse band matrix
        sparseA[0, :] = 0                                                       # Set up boundaries
        sparseA[-1, :] = 0                                                      
        sparseB = sparseA.T                                                     # Create second sparse band matrix
        f_x = 1/(2*dx1)*np.matmul(f, sparseB)                                   # Computes X term of the gradient
        f_y = 1/(2*dx2)*np.matmul(sparseA, f)                                   # Computes Y term of the gradient               
        return f_x[1:-1, 1:-1], f_y[1:-1, 1:-1]                                 # Returns gradient, sliced array
    def divergence(f_x, f_y, dx1, dx2):                                         # Compute the divergence using centered finit differences and ghost points: Returns one array (N1-2, N2-2)
        N = np.shape(f_x)[0]                                                    # Extract the shape of the mesh (squared)                                                  
        sparseA = -np.eye(N,N, -1)+np.eye(N,N,1)                                # Create first sparse band matrix
        sparseA[0, :] = 0                                                       # Set up boundaries
        sparseA[-1, :] = 0
        sparseB = sparseA.T                                                     # Create second sparse band matrix
        f_xx = 1/(2*dx1)*np.matmul(f_x, sparseB)                                # Compute X term of the divergence
        f_yy = 1/(2*dx2)*np.matmul(sparseA, f_y)                                # Compute Y term of the divergence
        ########### Intento con condiciones de frontera
        return (f_xx + f_yy)[1:-1,1:-1]                                         # Returns divergence, sliced array

    # In order to solve the lens equation we are going to use interpolation because:
    # 1. we don't wan't to create a model, we are going to reconstruct the model afterwards -> mantains unknown the parameters
    # 2. It ensures that all of the points of the simulated lens produce an effect on the images
    # 3. Diversity of numerical methods
    # We are going to use bicubic interpolation
    def interpolate2(x_inter, y_inter, field, x, y, dx1, dx2):                  # Bicubic interpolating function
        N = np.shape(field)[0]                                              # Extract the number of points of the array (Squared array)
        i0 = int(x_inter//dx1+(N//2)-1)                                     # Compute the index of the lower X bound of the interpolated coordinates
        j0 = int(y_inter//dx2+(N//2)-1)                                     # Compute the index of the lower y bound of the interpolated coordinates
        i1 = int(i0+1)                                                      # Compute the remaining indexes (4x4 points)
        j1 = int(j0+1)
        i2 = int(i0+2)
        j2 = int(j0+2)
        i3 = int(i0+3)
        j3 = int(j0+3)

        x0 = x[j0, i0]                                                      # Compute the coordinates of the corresponding indexes
        x1 = x[j0, i1]
        x2 = x[j0, i2]
        x3 = x[j0, i3]
        y0 = y[j0, i0]
        y1 = y[j1, i0]
        y2 = y[j2, i0]
        y3 = y[j3, i0]
        
        F = field[j0:j3+1, i0:i3+1]                                         # Extract the values of the field corresponding to the coordinates

        Y_coeff = np.zeros((4,4))                                           # Initialise the matrix of Y's
        X_coeff = np.zeros((4,4))                                           # Initialise the matrix of X's
        Y_coeff[0] = [1, 1, 1, 1]                                           # Create the Y matrix
        Y_coeff[1] = [y0, y1, y2, y3]
        Y_coeff[2] = [y0**2, y1**2, y2**2, y3**2]
        Y_coeff[3] = [y0**3, y1**3, y2**3, y3**3]

        X_coeff[0] = [1, 1, 1, 1]                                           # Create the X matrix
        X_coeff[1] = [x0, x1, x2, x3]
        X_coeff[2] = [x0**2, x1**2, x2**2, x3**2]
        X_coeff[3] = [x0**3, x1**3, x2**3, x3**3]

        X_inv = np.linalg.inv(np.array(X_coeff))                            # Compute the inverse of the X matrix
        Y_tinv = np.linalg.inv(np.array(Y_coeff).T)                         # Compute the inverse of the transpose of the Y matrix
        A = np.matmul(Y_tinv,np.matmul(F,X_inv))                            # Compute the matrix of coefficients
        X = np.array([1, x_inter, x_inter**2, x_inter**3])                  # Create the vector of the x coordinates to be interpolated
        Y = np.array([1, y_inter, y_inter**2, y_inter**3]).T                # Create the vector of the y coordinates to be interpolated

        return np.matmul(Y, np.matmul(A, X))                                # Return interpolation
    def root_find(p0,fx, fy, Jxx, Jxy, Jyx, Jyy, X1, X2, dx1, dx2):             # Newton's method for finding zeros
        k = 0                                                                   # Security parameter for maximum iteration
        epsilon = 0.00001                                                        # Resolution for the 0's of the function
        Jac = np.zeros((2,2))                                                   # Initialise the Jacobian
        F = np.ones((2,1))                                                      # Initialise the function
        p0 = np.array(p0)                                                       # Initialise the initial guess
        while((np.linalg.norm(F) >= epsilon)):                                  # Start iteration using the norm of the function as a criteria
            if(k <= 1000):                                                     # Set up the maximum iteration
                if((p0[0]<=X1[-1,0]) & (p0[0]>=X1[0,0]) & (p0[1]<=X2[0,-1]) & (p0[1]>=X2[0,0])):
                    Jac[0,0] = interpolate2(p0[0], p0[1], Jxx, X1, X2, dx1, dx2)    # Compute the Jacobian at the guess 
                    Jac[0,1] = interpolate2(p0[0], p0[1], Jxy, X1, X2, dx1, dx2) 
                    Jac[1,0] = interpolate2(p0[0], p0[1], Jyx, X1, X2, dx1, dx2) 
                    Jac[1,1] = interpolate2(p0[0], p0[1], Jyy, X1, X2, dx1, dx2) 
                    F[0] = interpolate2(p0[0], p0[1], fx, X1, X2, dx1, dx2)         # Compute the function at the guess
                    F[1] = interpolate2(p0[0], p0[1], fy, X1, X2, dx1, dx2) 
                    J_inv = np.linalg.inv(Jac)                                      # Invert the Jacobian
                    p0 = (p0 - np.matmul(F.T, J_inv.T)).flatten()
                                                         # Compute the next guess
            else:
                #    p0 = [None, None]
                break
            k +=1  
        return p0                                                               # Return final guess

    #################################################    MAIN    ############################################################
    # Considering the problem, is better to define the system in cartesian coordinates 
    # And operate with them in order to generalise the problem and them transform the systems into a suitable set of coordinates
    # The best would be to use elliptical coordinates because it can be used as a generalisation
    # Of the lensing potentials created by a unique body since the system must be almost isotropic

    star = source()                                                                                     # Build point source
    y1 = star.x                                                                                         # We extract the point coordinates
    y2 = star.y                                         
    lens1 = lens()                                                                                      # Build Lens
    iso = lens1.density_surf                                                                            # Build surface density over lens' mesh
    X1 = lens1.x1
    X2 = lens1.x2
    dx1 = lens1.dx1                                                                                     # We extract the dsteps
    dx2 = lens1.dx2                                                                                     # Same
    psi = poisson(2*iso, X1, X2, dx1, dx2)                                                              # Delta Psi = 2k = 2(Surface_density/Critical_surface_density)
    alpha_1, alpha_2 = grad(psi, dx1, dx2)
    X1 = X1[1:-1, 1:-1]                                                                                 # Rescale to the size of the derivative
    X2 = X2[1:-1, 1:-1]
    eq1 = y1 + alpha_1 - X1
    eq2 = y2 + alpha_2 - X2
    Jxx, Jxy = grad(eq1, dx1, dx2)                                                                      # We calculate the Jacobian
    Jyx, Jyy = grad(eq2, dx1, dx2)
    X1 = X1[1:-1, 1:-1]                                                                                 # Rescale to the size of the derivative
    X2 = X2[1:-1, 1:-1]
    eq1 = eq1[1:-1, 1:-1]                                                                               # Rescale to the size of the Derivative
    eq2 = eq2[1:-1, 1:-1]
    g = grad(eq1,dx1,dx2)[0]

    #plt.imshow(eq1)
    #plt.plot()
    pos = np.argwhere((abs(eq1) <= 0.03) & (abs(eq2) <= 0.03))                                          # Initial guesses for the arguments of the zeros of the lens equation
    zeros1 = X1[pos[:,0], pos[:,1]]                                                                     # We get the positions corresponding to the zeros
    zeros2 = X2[pos[:,0], pos[:,1]]
    roots = []                                                                                          # Array storing the roots of the equations
    for i in range(len(zeros1)):
        roots.append(root_find([zeros1[i], zeros2[i]], eq1, eq2, Jxx, Jxy, Jyx, Jyy, X1, X2, dx1, dx2)) # We find the zeros with Newton's method and Bicubic interpolation
    roots = np.array(roots)
    _, unique = np.unique(roots.round(6), axis = 0, return_index=True)                                  # For our purpuse we clear the array of repeated values that are equal up to certain decimals
    roots = roots[unique]
    #plt.scatter(roots[:,0],roots[:,1])
    #plt.show()
    physical_roots = roots*xi_0                                                                         # We rescale the lens positions to their physical values

    # Supposing that each node on the cell represents the central value of a pixel:
    N = np.shape(X1)[0]                                                                                 # We rescale the grid
    image = np.zeros((N-1, N-1))
    # We want to get the enclosing indexes of the roots
    for i in range(len(roots)):
        i0 = int(roots[i,0]//dx1+(N//2)-1)                                                              # Compute the index of the lower X bound of the interpolated coordinates
        j0 = int(roots[i,1]//dx2+(N//2)-1)                                                              # Compute the index of the lower y bound of the interpolated coordinates
        image[i0,j0] = 1
    ix1, ix2 = (X1[0,:-1]+1/2*dx1)*xi_0, (X2[:-1,0]+1/2*dx2)*xi_0                                       # We rescale the coordinates and center them in the pixels
    """    
    hf = h5py.File('Lens.hdf5', 'w')                                                                    # We save the file with the data
    hf.create_dataset("Image", data = image)
    coordinates = hf.create_group("Coordinates")
    coordinates.create_dataset("x1", data = ix1)
    coordinates.create_dataset("x2", data = ix2)
    source_parameters = hf.create_group("Source Parameters")
    source_parameters.attrs["x0"] = xs0*eta_0
    source_parameters.attrs["y0"] = ys0*eta_0
    source_parameters.attrs["z0"] = zs0
    lens_parameters = hf.create_group("Lens Parameters")
    lens_parameters.attrs["x0"] = xl0*xi_0
    lens_parameters.attrs["y0"] = yl0*xi_0
    lens_parameters.attrs["z0"] = zl0
    lens_parameters.attrs["Sigma"] = Sigma
    lens_parameters.attrs["xc"] = xc*xi_0
    lens_parameters.attrs["f"] = f
    system_parameters = hf.create_group("System Parameters")
    system_parameters.attrs["D_S"] = D_s
    system_parameters.attrs["D_L"] = D_l
    system_parameters.attrs["D_LS"] = D_ls
    system_parameters.attrs["Xi0"] = xi_0
    system_parameters.attrs["Eta0"] = eta_0
    system_parameters.attrs["Alpha_scale"] = alpha_scale
    hf.close() # Cerramos el archivo
    """
x = np.random.rand()
y = np.random.rand()
Sigma = np.random.rand()+0.01    # Cannot be 0        
f = np.random.rand()+0.001           # Cannot be 0
Generate(x, y, Sigma, f)
