from src.calculus import *
from src.construct import lens
import matplotlib.pyplot as plt
import h5py

# Test must be done with sigma = 2.5 f = 1 centered model, crit radius 0.05
flens = h5py.File('/Users/bach/Desktop/Lensing/Lensing/output/Lens_test.hdf5', 'r')                      # Open the file (Change to your path)
image = np.array(flens["Image"])                                                                     # We get the image
coordinates = flens["Coordinates"]                                                                   # Get the coordinates and store them
x1_image = np.array(coordinates["x1"])
x2_image = np.array(coordinates["x2"])
D_L = flens["System Parameters"].attrs["D_L"]                                                        # We get the distance to the lens
D_S = flens["System Parameters"].attrs["D_S"]                                                        # We get the distance to the source
D_LS = flens["System Parameters"].attrs["D_LS"]                                                      # We get the difference
xc = flens["Lens Parameters"].attrs["xc"]

def test_gradient():
    epsilon = 1
    x = np.linspace(-5, 5, 2000)
    y = np.linspace(-5, 5, 2000)
    dx1, dx2 = (np.max(x)-np.min(x))/x.size, (np.max(y)-np.min(y))/y.size
    X, Y = np.meshgrid(x, y)
    field = (X**2)*(Y**2)
    X_test = X[1:-1,1:-1]
    Y_test = Y[1:-1,1:-1]
    field_test_x = 2*X_test*(Y_test**2)
    field_test_y = 2*Y_test*(X_test**2)
    gradient = grad(field, dx1, dx2)
    gradient1 = gradient[0]
    gradient2 = gradient[1]
    assert np.all((gradient1 <= field_test_x+epsilon) & (gradient1 >= field_test_x-epsilon))
    assert np.all((gradient2 <= field_test_y+epsilon) & (gradient2 >= field_test_y-epsilon))


def test_divergence():
    epsilon = 1
    x = np.linspace(-5, 5, 2000)
    y = np.linspace(-5, 5, 2000)
    dx1, dx2 = (np.max(x)-np.min(x))/x.size, (np.max(y)-np.min(y))/y.size
    X, Y = np.meshgrid(x, y)
    field_x = (X**2)*(Y**2)
    field_y = (X**2)+(Y**2)
    X_test = X[1:-1,1:-1]
    Y_test = Y[1:-1,1:-1]
    field_test = 2*X_test*(Y_test**2)+2*Y_test
    div = divergence(field_x, field_y, dx1, dx2)
    assert np.all((div <= field_test+epsilon) & (div >= field_test-epsilon))


def test_zeros():
    epsilon = 0.0001
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    dx1, dx2 = (np.max(x)-np.min(x))/x.size, (np.max(y)-np.min(y))/y.size
    X, Y = np.meshgrid(x, y)  
    fx = Y*X**2
    fy = X*Y**2 
    Jxx = 2*X*Y
    Jxy = X**2
    Jyx = Y**2
    Jyy = 2*Y*X
    p0 = [0.1, 0.1]
    roots = root_find(p0, fx, fy, Jxx, Jxy, Jyx, Jyy, X, Y, dx1, dx2)
    assert np.all((roots <= epsilon) & (roots >= -epsilon))

def test_interpolate2():
    epsilon = 0.000001
    x = np.linspace(-5, 5, 200)
    y = np.linspace(-5, 5, 200)
    dx1, dx2 = (np.max(x)-np.min(x))/x.size, (np.max(y)-np.min(y))/y.size
    X, Y = np.meshgrid(x, y)  
    func = (X**3)*(Y**3)    
    p0 = [1.43566, 2.4355]
    func_test = (p0[0]**3)*(p0[1]**3)
    inter = interpolate2(p0[0], p0[1], func, X, Y, dx1, dx2)
    assert ((func_test <= inter + epsilon) &  (func_test >= inter - epsilon))

def test_poisson():
    epsilon = 2
    lens1 = lens()
    surf_dens = lens1.density_surf
    X1, X2 = lens1.x1, lens1.x2 
    dx1, dx2 = X1[0,1]-X1[0,0], X2[1,0]-X2[0,0]
    psi = poisson(surf_dens, X1, X2, dx1, dx2) 
    alpha_1 = grad(psi, dx1, dx2)[0]
    alpha_2 = grad(psi, dx1, dx2)[1]
    alpha = np.sqrt(alpha_1**2+alpha_2**2)
    # We are going to multiply for X in order to avoid the singularities in the analytical solution
    R = np.sqrt(X1**2+X2**2)[1:-1,1:-1]
    alpha_R = R*alpha
    sol = np.sqrt((xc**2)+(R**2)) - xc
    # Every number of cells must be odd
    assert (np.all(alpha_R <= sol + epsilon) and np.all(alpha_R >= sol - epsilon))