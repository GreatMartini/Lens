from src.statfunc import *
#### Hacer tests con caso trivial de prueba -> generar caso trivial ###########
def test_surf():
    sig_calc = surf_dens(1, 1, 0, 0, 1)
    assert sig_calc == 1/(2*G)
def test_map_source():
    x1, x2 = 1, 1
    alpha1, alpha2 = 1, 1
    sources = map_source(x1, x2, alpha1, alpha2)
    assert np.all(sources) == D_S/D_L-D_LS
    # Con einstein ring y potencial normal:
def test_map_lens():
    # Resolviendo primero angulos con potencial dados puntos nulos
    # angulo deberia ser 0 => eq==0
    x1, x2 = 1, 1
    y1, y2 = 1, 1
    alpha1, alpha2 = 0, 0
    eqs = map_lens(y1, y2, x1, x2, alpha1, alpha2)
    assert np.all(eqs) == 0

#### Arreglar lens solve
def test_cost_function():
    # Verificar cost_function porque con estos puntos da 2.06
    obs_pos = [[1, 1]]
    calculated_pos = [[1, 1]]
    cost = cost_function(obs_pos, calculated_pos)
    print(cost)
    assert cost == 0
def test_log_prior():
    # Testear con randoms
    theta = [2.5, 0.5]
    assert log_prior(theta) != -np.inf
    theta = [5.1, 0]
    assert log_prior(theta) == -np.inf
def test_log_likelihood():
    assert log_likelihood(2) == -1
def test_log_posterior():
    assert log_posterior(1,1) == 2
def test_acceptance():    
    assert acceptance(1, 0) == 0
def test_alpha_test():
    theta_prime = "theta_prime"
    theta = "theta"
    assert alpha_test(0, theta_prime, theta) == theta_prime
    assert alpha_test(-100000, theta_prime, theta) == theta
def test_theta_prime():
    assert np.all(theta_prime([1000,1000])) != 0 