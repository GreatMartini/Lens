import numpy as np
import h5py
from gen_params import G, c

lens = h5py.File("../output/Lens1.hdf5", 'r')                                                                  # Open the file
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
lens.close()
dens_critic = c**2/(4*np.pi*G)*D_S/(D_L*D_LS)
# We want to extract the images' coordinates:
coord = np.argwhere(image == 1)                                                                      # We extract the coordinates
images_x1 = x1_image[coord[:,1]]                                                                     # Because the first axis changes over the columns
images_x2 = x2_image[coord[:,0]]
images = np.array([x1_image[coord[:,1]], x2_image[coord[:,0]]]).T

################################################################################
# Now that we have the coordinates we want to create a lens' mesh that is suffieciently big to avoid going out of it
# Theoretically we would have to calculate it using xi_0 but we are going to do it arbitrarily for now 
# The image has the usual cartesian coordinates, since we trim the arrays we need to rescale the arguments too
# Which means that we will have to flip the image in order to find the positions
# Since the mesh in y is negative at the top and positive at the bottom
# Against intuition, x corresponds to the columns and y to the rows