from src.calculus import *

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
def test_interpolate2():
    epsilon = 0.1
    x = np.linspace(-5, 5, 2000)
    y = np.linspace(-5, 5, 2000)
    dx1, dx2 = (np.max(x)-np.min(x))/x.size, (np.max(y)-np.min(y))/y.size
    X, Y = np.meshgrid(x, y)
    field = (X**3)*(Y**3)
    x_0, y_0 = 2.546, 1.457
    field_0 =  (x_0**3)*(y_0)**3
    field_inter = interpolate2(x_0, y_0, field, X, Y, dx1, dx2)
    assert np.all((field_inter <= field_0+epsilon) & (field_inter >= field_0-epsilon))
def test_zeros():
    epsilon = 0.0001
    x = np.linspace(-5, 5, 2000)
    y = np.linspace(-5, 5, 2000)
    dx1, dx2 = (np.max(x)-np.min(x))/x.size, (np.max(y)-np.min(y))/y.size
    X, Y = np.meshgrid(x, y)  
    fx = X**2
    fy = Y**2 
    Jxx = 2*X
    Jxy = np.zeros((np.shape(X)[0],np.shape(X)[1]))
    Jyx = np.zeros((np.shape(X)[0],np.shape(X)[1]))
    Jyy = 2*Y
    p0 = [0.1, 0.1]
    roots = root_find(p0,fx, fy, Jxx, Jxy, Jyx, Jyy, X, Y, dx1, dx2)
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
