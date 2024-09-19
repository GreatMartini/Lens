""" Module that reads the parameters of the image file 

Parameters:
-----------
lens : hdf5
    Image file.
image : 2darray
    Array containing 1's and 0's indicating where the images are formed.
coordinates : hdf5 group
    Contains the image's coordinates.
x1_image, x2_image : 2darrays
    Coordinates of the semi-major and semi-minor axes respectively.
D_L : float
    Distance to the lens plane.
D_S :float
    Distance to the source plane.
D_LS : float
    Distance between the lens and the source plane.
xc : float
    Critical radius of the non-singulat isothermal potential.
dens_critic : float
    Critical surface density for the reduction of the non-singular isothermal surface density.
coord : int
    Argument of the positions of the formed images.                                                                      
images_x1 : 1darray
    Contains the semi-major axis coordinates of the formed images.                                                                     
images_x2 : 1darray
    Contains the semi-minor axis coordinates of the formed images.
images : 2darray
    Contains the coordinates of the formed images.
    
"""
import numpy as np
import h5py
from .gen_params import G, c

lens = h5py.File('/Users/bach/Desktop/Lensing/Lensing/output/Lens_test.hdf5', 'r')                      # Open the file (Change to your path)
image = np.array(lens["Image"])                                                                     # We get the image
coordinates = lens["Coordinates"]                                                                   # Get the coordinates and store them
x1_image = np.array(coordinates["x1"])
x2_image = np.array(coordinates["x2"])
D_L = lens["System Parameters"].attrs["D_L"]                                                        # We get the distance to the lens
D_S = lens["System Parameters"].attrs["D_S"]                                                        # We get the distance to the source
D_LS = lens["System Parameters"].attrs["D_LS"]                                                      # We get the difference
xc = lens["Lens Parameters"].attrs["xc"]

# we print in order to confirm the results
#print(lens["Lens Parameters"].attrs["Sigma"])
#print(lens["Lens Parameters"].attrs["f"])
lens.close()
dens_critic = c**2/(4*np.pi*G)*D_S/(D_L*D_LS)
# We want to extract the images' coordinates:
coord = np.argwhere(image == 1)                                                                      # We extract the coordinates
images_x1 = x1_image[coord[:,1]]                                                                     # Because the first axis changes over the columns
images_x2 = x2_image[coord[:,0]]
# Revisarrrrrrr!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:::::::::::::::::::
images = np.array([x1_image[coord[:,1]], x2_image[coord[:,0]]]).T

################################################################################
# Now that we have the coordinates we want to create a lens' mesh that is suffieciently big to avoid going out of it
# Theoretically we would have to calculate it using xi_0 but we are going to do it arbitrarily for now 
# The image has the usual cartesian coordinates, since we trim the arrays we need to rescale the arguments too
# Which means that we will have to flip the image in order to find the positions
# Since the mesh in y is negative at the top and positive at the bottom
# Against intuition, x corresponds to the columns and y to the rows