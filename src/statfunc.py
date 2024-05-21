############################## Proper to reconstruction #####################################
from calculus import *
from rec_params import *
from scipy.stats import uniform

def surf_dens(sigma, f, x1, x2, xc):                                        # surface density for the iso potential
    return (sigma**2)/(2*G)*np.sqrt(f)/np.sqrt(x1**2 + (f*x2)**2 + xc**2)
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