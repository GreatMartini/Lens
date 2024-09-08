import numpy as np
import matplotlib.pyplot as plt
import h5py
lens = h5py.File("/Users/bach/Desktop/Lensing/Lensing/output/Lens_test.hdf5", 'r')
image = lens["Image"]

xi_0 = lens["System Parameters"].attrs["Xi0"]
eta_0 = lens["System Parameters"].attrs["Eta0"]
y1 = lens["Source Parameters"].attrs["x0"]/eta_0
y2 = lens["Source Parameters"].attrs["y0"]/eta_0

Ry = np.sqrt(y1**2+y2**2)
x1 = lens["Coordinates"]["x1"]/xi_0
x2 = lens["Coordinates"]["x2"]/xi_0
xc = lens["Lens Parameters"].attrs["xc"]/lens["System Parameters"].attrs["Xi0"]
rayon_einstein = np.sqrt(1-2*xc)
print(rayon_einstein)
plt.imshow(image, cmap = "gray", extent = [np.min(x1), np.max(x1), np.min(x2), np.max(x2)])
plt.show()