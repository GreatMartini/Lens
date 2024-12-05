""" Module that contains the fuctions proper to the mass distribution reconstruction algorithm"""
from .calculus import *
from .rec_params import *
from scipy.stats import uniform

def surf_dens(sigma, f, x1, x2, xc):
    """  Computes the non-reduced isothermal surface density. 
    
    Parameters:
    ------------
    
    sigma : float
        Dispersion velocity of the isothermal surface density.
    f : float
        Step size along the semi-major axis.
    x1 : 2darray
        Mesh coordinates of the semi-major axis.
    x2 : 2darray
        Mesh coordinates of the semi-minor axis.
    xc : float
        Critical radius of the isothermal potential.
    
    Returns:
    --------

    surf_dens : 2darray
        Non-reduced isothermal surface density.

    """                                       
    return (sigma**2)/(2*G)*np.sqrt(f)/np.sqrt(x1**2 + (f*x2)**2 + xc**2)


def map_source(x1, x2, alpha_1, alpha_2):
    """  Maps the images on the lens plane to the source plane. 
    
    Parameters:
    ------------

    x1 : 1darray
        Image coordinates of the semi-major axis.
    x2 : 1darray
        Image coordinates of the semi-minor axis.
    alpha_1 : 1darray
        Semi-major axis components of the deviation angle evaluated at the image positions.
    alpha_2 : 1darray
        Semi-minor axis components of the deviation angle evaluated at the image positions.
    
    Returns:
    --------

    mean(eta1), mean(eta2) : floats
        Reeduced mean coordinates of the predicted source.

    """        
    eta1 = x1*D_S/D_L - alpha_1*D_LS
    eta2 = x2*D_S/D_L - alpha_2*D_LS
    return np.mean(eta1), np.mean(eta2)


def map_lens(y1, y2, x1, x2, alpha_1, alpha_2):
    """  Lens equations, we want them to be 0 at the positions of the formed images.
    
    Parameters:
    ------------

    x1 : 2darray
        Mesh coordinates of the semi-major axis.
    x2 : 2darray
        Mesh coordinates of the semi-minor axis.
    y1 : float
        Semi-major axis coordinate of the source.
    y2 : float
        Semi-minor axis coordinate of the source.
    alpha_1 : 2darray
        Semi-major axis components of the deviation angle evaluated over the grid.
    alpha_2 : 2darray
        Semi-minor axis compoinets of the deviation angle evaluated over the grid.
    
    Returns:
    --------

    eq1, eq2 : 2darrays
        Components of the lens equation evaluated over the grid.

    """
    eq1 = y1 - x1 + alpha_1
    eq2 = y2 - x2 + alpha_2
    return eq1, eq2


def lens_solve(theta_init, X1, X2, dx1, dx2, xc):
    """  Solves the lens equation with the specified parameters.
    
    Parameters:
    ------------

    theta_init: 1darray
        Velocity dispersion and ellipticity of the lens.
    X1 : 2darray
        Mesh coordinates of the semi-major axis.
    X2 : 2darray
        Mesh coordinates of the semi-minor axis.
    alpha_1 : 2darray
        Semi-major axis components of the deviation angle evaluated over the grid.
    alpha_2 : 2darray
        Semi-minor axis compoinets of the deviation angle evaluated over the grid.
    dx1 : float
        Step in the direction of the semi-major axis.
    dx2 : float
        Step in the direction of the semi-minor axis.  
    xc : float
        Critical radius of the isothermal potential. 
    
    Returns:
    --------

    roots : 2darray
        Contains the roots of the lens equations.

    """
    dens = surf_dens(theta_init[0], theta_init[1], X1, X2, xc)/dens_critic                              # Surface density of the potential
    pot = poisson(2*dens, X1, X2, dx1, dx2)                                                             # Compute the potential
    alpha_1, alpha_2 = grad(pot, dx1, dx2)                                                              # Compute the deviation angles
    X1, X2 = X1[1:-1, 1:-1], X2[1:-1, 1:-1]                                                             # Rescale the grid
    a1 = np.zeros(images_x1.size)                                                                       # Values of the first component of the deviation angle at the images' positions
    a2 = np.zeros(images_x2.size)                                                                       # Values of the second component of the deviation angle at the images' positions
    for i in range(images_x1.size):                                                                     # We interpolate the values of the deviation angle in order to find its values at the images' positions
        a1[i] = interpolate2(images_x1[i], images_x2[i], alpha_1, X1, X2, dx1, dx2)
        a2[i] = interpolate2(images_x1[i], images_x2[i], alpha_2, X1, X2, dx1, dx2)
    y1, y2 = map_source(images_x1, images_x2, a1, a2)                                                   # We map to the source
    eq1, eq2 = map_lens(y1, y2, X1, X2, alpha_1, alpha_2)                                               # We map back to the lens plane

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
    # We need to solve the problem when there are no roots or the root finding algorithm diverges. 
    #if (roots.size > 0):
    #    roots = roots[roots[:,0] != None]
    #else:
    #    roots = np.array([[0, 0],[0,0]])
    return roots


def cost_function(lens_pos, calculated_pos):
    """  Computes the cost function in function of the predicted and real positions of the images.
    The cost function is evaluated with the norm of the distance between the real and predicted
    positions of the images. In further development we could implement two cost functions one for each component
    of the distances.
    
    Parameters:
    ------------

    lens_pos : 2darray
        Coordinates of the observed images.
    calculated_pos : 2darray
        Coordinates of the predicted images.
    
    Returns:
    --------

    Chi2 : float
        Cost function.

    """
    calculated_pos = np.sort(calculated_pos, axis=0)        # We sort
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
        if ((i0 < x_pos.size) & (j0 < y_pos.size) & (i0 >=0) & (j0 >=0)):                                                                                                                                                       # Compute the index of the lower y bound on the grid of the image of the solutions
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
    return Chi2

    
def log_prior(theta):
    """  Computes the logarithm of the prior. 
    
    Parameters:
    ------------
    
    theta : 1darray
        The dispersion velocity and ellipticity.
    
    Returns:
    --------

    surf_dens : 2darray
        Non-reduced isothermal surface density.
    prior : float
        Joint prior distribution for the two parameters

    """     
    prior1 = uniform.pdf(theta[0],0,5)
    prior2 = uniform.pdf(theta[1],0.00001, 1.0000)
    if ((prior1!=0) & (prior2!=0)):
        return np.log(prior1)+np.log(prior2)
    else:
        return -np.inf


def log_likelihood(chi2):
    """  Computes the logarithm of the likelihood. 
    
    Parameters:
    ------------
    
    chi2 : float
        Cost function.
    
    Returns:
    --------

    log_likelihood : float
        Logarithm of the likelihood

    """    
    return -1.0*chi2/2


def log_posterior(like, prior):
    """  Computes the logarithm of the posterior distribution. 
    
    Parameters:
    ------------
    
    like : float
        Logarithm of the likelihood.
    prior : float
        Logarithm of the prior distribution.
    
    Returns:
    --------

    log_posterior : float
        Logarithm of the posterior distribution.
    """  

    return like+prior


def acceptance(post_prime, post):
    """  Computes the accetance. 
    
    Parameters:
    ------------
    
    post_prime : float
        Logarithm of the posterior distribution of the second set of parameters
    post : float
        Logarithm of the posterior distribution of the first set of parameters
    
    Returns:
    --------

    acc : float
        Difference between the posteriors (logarithm of the acceptance)
    """  
    if (post_prime - post < 0):
        return post_prime - post
    else:
        return 0


def alpha_test(acc, theta_prime, theta):
    """  Computes the accetance test. 
    
    Parameters:
    ------------

    acc : float
        Logarithm of the acceptance.
    theta_prime : 1darray
         Second set of parameters
    theta : 1darray
        First set of parameters
    
    Returns:
    --------

    params : 1darray
        The set of parameters most likely to be the true ones based on the acceptance test.

    """  
    logu = np.log(1-np.random.uniform(0,1))
    # Si el ratio es mas grande que 1 tengo que elegir la acceptancia en 1
    if(acc >= logu):
        return theta_prime
    else:
        return theta


def theta_prime(theta):
    """  Computes the accetance test. 
    
    Parameters:
    ------------

    theta : 1darray
        First set of parameters
    
    Returns:
    --------

    t : 1darray
        Second set of parameters obtained from the normal distribution centered on the first set of parameters.

    """  
    sigma = [0.5, 0.1]
    t = np.random.normal(theta, sigma)
    return t 