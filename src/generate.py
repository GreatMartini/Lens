################################### Script that generates the image data ##################################
from construct import *
from calculus import *
import h5py


def Generate(xs0, ys0, Sigma, f, save = True):
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
    if (save == True):    
        hf = h5py.File('../output/Lens1.hdf5', 'w')                                                                    # We save the file with the data
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

