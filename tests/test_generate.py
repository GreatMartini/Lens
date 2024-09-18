from src.generate import *
from src import gen_params
def test_gen():
    epsilon = 0.09
    #Generate(0.5, 0.5, 1, 0, save = True)
    lens = h5py.File("/Users/bach/Desktop/Lensing/Lensing/output/Lens_test.hdf5", 'r')
    image = lens["Image"]

    xi_0 = lens["System Parameters"].attrs["Xi0"]
    eta_0 = lens["System Parameters"].attrs["Eta0"]
    y1 = lens["Source Parameters"].attrs["x0"]/eta_0
    y2 = lens["Source Parameters"].attrs["y0"]/eta_0
    Ry = np.sqrt(y1**2+y2**2)
    x1 = lens["Coordinates"]["x1"]/xi_0
    x2 = lens["Coordinates"]["x2"]/xi_0
    X1, X2 = np.meshgrid(x1, x2)
    location = np.argwhere(image == np.max(image))
    coeffs = [1, 2*Ry, Ry**2+2*gen_params.xc-1,2*gen_params.xc*Ry]
    solution = np.abs(np.roots(coeffs))
    pos1 = []
    pos2 = []
    for i in range(len(location)):
        pos1.append(X1[location[i,0], location[i,1]])
        pos2.append(X2[location[i,0], location[i,1]])
    pos = np.sqrt(np.array(pos1)**2 + np.array(pos2)**2)
    pos = np.unique(pos)
    solution = np.unique(solution)
    for i in range(len(pos)-len(solution)): #?
        solution = np.append(solution, solution[1])
    solution.sort()
    pos.sort()
    for i in range(len(pos)):
        assert ((pos[i] <= solution[i] + epsilon) & (pos[i] >= solution[i] - epsilon))