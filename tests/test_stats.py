from src.statfunc import *

def test_surf():
    sig_calc = surf_dens(1, 1, 0, 0, 1)
    assert sig_calc == 1/(2*G)

#def test_map_source():
# Con einstein ring y potencial normal:
