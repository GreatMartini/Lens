# Tests for the Generation script
import pytest
import numpy as np
import sys
sys.path.insert(1, '/Users/bach/Desktop/Lensing/Lensing_clean/')
from fun import divergence

def test_divergence():
    dstep = 50
    x, y = np.linspace(-5,5,dstep), np.linspace(-5,5,dstep)
    X, Y = np.meshgrid(x, y)
    Fx = X**2
    Fy = Y**2
    A = divergence(Fx,Fy,dstep,dstep)
    X, Y = X[1:-1,1:-1], Y[1:-1, 1:-1]
    print(A)
    print(2*X+2*Y)
    assert np.any(A == 2*X+2*Y)