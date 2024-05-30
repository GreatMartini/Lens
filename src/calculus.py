############################### Mathematical functions for computations, interpolations and root finding ############################
from scipy.fft import fftn, ifftn
from .gen_params import *


def poisson(field, x1, x2, dx1, dx2):                                       # Solves the Poisson equation using Fourier Transform and zero padding
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
    ft = g/K_squared                                                        # Compute the laplacian in the fourier space
    f = np.real(ifftn(ft))[Npad2:-Npad2,Npad1:-Npad1]                       # Returns the solution of the poisson equation, slices the padding
    return -f
def grad(f, dx1 , dx2):                                                       # Computes the gradient using centered finite differences and ghost points: Returns two arrays (N1-2, N2-2)
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
    epsilon = 0.00000001                                                        # Resolution for the 0's of the function
    Jac = np.zeros((2,2))                                                   # Initialise the Jacobian
    F = np.ones((2,1))                                                      # Initialise the function
    p0 = np.array(p0)                                                       # Initialise the initial guess
    while((np.linalg.norm(F) >= epsilon)):                                  # Start iteration using the norm of the function as a criteria
        if(k <= 1000):                                                     # Set up the maximum iteration
            if((p0[0]<=np.max(X1)) & (p0[0]>=np.min(X1)) & (p0[1]<=np.max(X2)) & (p0[1]>=np.min(X2))):
                Jac[0,0] = interpolate2(p0[0], p0[1], Jxx, X1, X2, dx1, dx2)    # Compute the Jacobian at the guess 
                Jac[0,1] = interpolate2(p0[0], p0[1], Jxy, X1, X2, dx1, dx2) 
                Jac[1,0] = interpolate2(p0[0], p0[1], Jyx, X1, X2, dx1, dx2) 
                Jac[1,1] = interpolate2(p0[0], p0[1], Jyy, X1, X2, dx1, dx2) 
                F[0] = interpolate2(p0[0], p0[1], fx, X1, X2, dx1, dx2)         # Compute the function at the guess
                F[1] = interpolate2(p0[0], p0[1], fy, X1, X2, dx1, dx2) 
                J_inv = np.linalg.inv(Jac)                                      # Invert the Jacobian
                p0 = (p0 - np.matmul(F.T, J_inv.T)).flatten()
        else:
            break
        k +=1
    return p0                                                               # Return final guess
