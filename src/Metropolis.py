import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.fft import fftn, ifftn
from scipy.stats import uniform


G, c = 1, 1                                                                                         # Natural units
lens = h5py.File("Lens.hdf5", 'r')                                                                  # Open the file
image = np.array(lens["Image"])                                                                     # We get the image
coordinates = lens["Coordinates"]                                                                   # Get the coordinates and store them
x1_image = np.array(coordinates["x1"])
x2_image = np.array(coordinates["x2"])
D_L = lens["System Parameters"].attrs["D_L"]                                                                           # We get the distance to the lens
D_S = lens["System Parameters"].attrs["D_S"]                                                                           # We get the distance to the source
D_LS = lens["System Parameters"].attrs["D_LS"]                                                                         # We get the difference
xc = lens["Lens Parameters"].attrs["xc"]

# we print in order to confirm the results
print(lens["Lens Parameters"].attrs["Sigma"])
print(lens["Lens Parameters"].attrs["f"])

dens_critic = c**2/(4*np.pi*G)*D_S/(D_L*D_LS)
# We want to extract the images' coordinates:
coord = np.argwhere(image == 1)                                                                      # We extract the coordinates
images_x1 = x1_image[coord[:,1]]                                                                     # Because the first axis changes over the columns
images_x2 = x2_image[coord[:,0]]
images = np.array([x1_image[coord[:,1]], x2_image[coord[:,0]]]).T

def surf_dens(sigma, f, x1, x2, xc):
    return (sigma**2)/(2*G)*np.sqrt(f)/np.sqrt(x1**2 + (f*x2)**2 + xc**2)
def poisson(field, x1, x2, dx1, dx2):                                       # Solves the Poisson equation using Fourier Transform and zero padding
    Ntot1 = np.shape(x1)[1]                                                 # We define the total number of points of the array                                                 
    Ntot2 = np.shape(x2)[0]
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

########################## ARREGLAR #########################################
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
    return p0   
######################################################################
# Since we did not adimensionated the lens equation
# The lens equation is now
# D_L/D_S*eta = Xi - alpha*D_LS*D_L/D_S
def map_source(x1, x2, alpha_1, alpha_2):
    eta1 = x1*D_S/D_L - alpha_1*D_LS
    eta2 = x2*D_S/D_L - alpha_2*D_LS
    return np.mean(eta1), np.mean(eta2)
def map_lens(y1, y2, x1, x2, alpha_1, alpha_2):
    eq1 = y1 -x1*D_S/D_L + alpha_1*D_LS
    eq2 = y2 -x2*D_S/D_L + alpha_2*D_LS
    return eq1, eq2
#################################### Quitar parches despues de arreglar ###################################
def lens_solve(theta_init, X1, X2, dx1, dx2, xc):
    dens = surf_dens(theta_init[0], theta_init[1], X1, X2, xc)/dens_critic                              # Surface density of the potential
    pot = poisson(2*dens, X1, X2, dx1, dx2)
    alpha_1, alpha_2 = grad(pot, dx1, dx2)
    X1, X2 = X1[1:-1, 1:-1], X2[1:-1, 1:-1]
    # Now we have to interpolate the values in alpha
    a1 = np.zeros(images_x1.size)
    a2 = np.zeros(images_x2.size)
    for i in range(images_x1.size):
        a1[i] = interpolate2(images_x1[i], images_x2[i], alpha_1, X1, X2, dx1, dx2)
        a2[i] = interpolate2(images_x1[i], images_x2[i], alpha_2, X1, X2, dx1, dx2)
    y1, y2 = map_source(images_x1, images_x2, a1, a2)
    eq1, eq2 = map_lens(y1, y2, X1, X2, alpha_1, alpha_2)

    Jxx, Jxy = grad(eq1, dx1, dx2)                                                                      # We calculate the Jacobian
    Jyx, Jyy = grad(eq2, dx1, dx2)
    X1 = X1[1:-1, 1:-1]                                                                                 # Rescale to the size of the derivative
    X2 = X2[1:-1, 1:-1]
    eq1 = eq1[1:-1, 1:-1]                                                                               # Rescale to the size of the Derivative
    eq2 = eq2[1:-1, 1:-1]
    pos = np.argwhere((abs(eq1) <= 0.08) & (abs(eq2) <= 0.08))
    zeros1 = X1[pos[:,0], pos[:,1]]                                                                     # We get the positions corresponding to the zeros
    zeros2 = X2[pos[:,0], pos[:,1]]
    roots = []                                                                                          # Array storing the roots of the equations
    for i in range(len(zeros1)):
        roots.append(root_find([zeros1[i], zeros2[i]], eq1, eq2, Jxx, Jxy, Jyx, Jyy, X1, X2, dx1, dx2)) # We find the zeros with Newton's method and Bicubic interpolation
    roots = np.array(roots)
    if (roots.size > 0):
        roots = roots[roots[:,0] != None]
    else:
        roots = np.array([[0, 0],[0,0]])
    return roots

def cost_function(lens_pos,calculated_pos):
    #  calculated_pos = np.sort(calculated_pos, axis=0)        # We sort so that we can identify de shifts
    x_pos = calculated_pos[:,0]             # Extract x
    y_pos = calculated_pos[:,1]             # Extract y
    # We are going to rescale the calculate images coordinates to the pixel's centers
    # In order to do so we call the coordinates of the images and the delta_step
    # The delta step will be afterwards taken as the error in the position
    dx1_image = x1_image[1]-x1_image[0] # We calculate the delta _step of the image
    dx2_image = x2_image[1]-x2_image[0] # We calculate the delta _step of the image
    N = x1_image.size
    for i in range(len(x_pos)):
        i0 = int(x_pos[i]//dx1_image+(N//2)-1)                                                              # Compute the index of the lower X bound in the grid of the image of the solutions
        j0 = int(y_pos[i]//dx2_image+(N//2)-1)
        if ((i0 < x_pos.size) & (j0 < y_pos.size) & (i0 >=0) & (j0 >=0)):                                                              # Un poco inexacto                                                                                                                 # Compute the index of the lower y bound on the grid of the image of the solutions
            if (abs(x_pos[i]-x1_image[i0]) <= abs(x_pos[i]-(x1_image[i0]+dx1_image))):                          # Compute the distances to the grid points and select the closest node
                x_pos[i] = x1_image[i0]
            else:
                x_pos[i] = x1_image[i0]+dx1_image
            if (abs(y_pos[i]-x2_image[j0]) <= abs(y_pos[i]-(x2_image[j0]+dx2_image))):                        # Compute the distances to the grid points and select the closest node
                y_pos[i] = x2_image[j0]
            else:
                y_pos[i] = x2_image[j0]+dx2_image
            
    pos = np.concatenate((x_pos[:,None], y_pos[:,None]), axis = 1)   # We create an array of the position                              

    pos = np.unique(pos, axis = 0)                                                                      # We remove the duplicates
    N_ima = np.shape(pos)[0]                                                                            # That gives us the number of images        
    # Separar Nima de Chi2
    # We now have to find the closest points.
    # We now set the error to the surface of the pixels
    err = dx1_image*dx2_image # As the error is constant we don't have to take it into account
    err2 = err*err

    # We are going to find the closest images to the predicted ones
    obs_images = np.concatenate((images_x1[:,None], images_x2[:,None]), axis = 1)
    distances_squared = np.zeros((np.shape(obs_images)[0], np.shape(pos)[0]))

    for i in range(len(distances_squared)):
        distances_squared[i] = (obs_images[i,0]-pos[:,0])**2+(obs_images[i,1]-pos[:,1])**2
    Chi2 = np.sum(np.min(distances_squared, axis = 0))
    # Chi2 esta siendo utilaza con la norma,despues habrá que reemplazar un chi2 para cada eje
    return Chi2

################################################################################
    
    # For our purpose we are not going to use the error sigma becaus due to the root finding function, the error is too small (<e-18)
    # That will blow up the calculation, so we are going only to minimize distances taking sigma = 1
    # We could repeat the calculations with sigma without using the zero finding functions
    # Also it will be maybe an error due to the pixels's size but it is constant for all the images so it is not relevant for the minimization scheme
def log_prior(theta):
    prior1 = uniform.pdf(theta[0],0,5.5)
    prior2 = uniform.pdf(theta[1],0.00001, 1.00001)
    if ((prior1!=0) & (prior2!=0)):
        return np.log(prior1)+np.log(prior2)
    else:
        return -np.inf
def log_likelihood(chi2):
    return -1.0*chi2/2
def log_posterior(like, prior):
    return like+prior
def acceptance(post_prime, post):
    return post_prime-post
def test(acc, theta_prime, theta):
    logu = np.log(np.random.uniform(0,1))
    if(acc >= logu):
        return theta_prime
    else:
        return theta
def theta_prime(theta):
    sigma = [0.5, 0.1]
    t = np.random.normal(theta, sigma)
    return t 
################################################################################
# Now that we have the coordinates we want to create a lens' mesh that is suffieciently big to avoid going out of it
# Theoretically we would have to calculate it using xi_0 but we are going to do it arbitrarily for now 
# The image has the usual cartesian coordinates, since we trim the arrays we need to rescale the arguments too
# Which means that we will have to flip the image in order to find the positions
# Since the mesh in y is negative at the top and positive at the bottom
# Against intuition, x corresponds to the columns and y to the rows
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
