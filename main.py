import os
import sys
from src.calculus import *
from src.construct import *
from src.gen_params import *
from src.generate import *
from src.rec_params import *
from src.reconstruct import *
from src.statfunc import *

print("If you only want to generate new image enter G, if you only want to reconstruct the lens enter R if you want to do both enter GR:")
sys.argv[0] = input()
if (sys.argv[0] == "G"):
    x = 0
    y = 0
    Generate(x, y, Sigma, f)
elif (sys.argv[0] == "R"):
    Metropolis()
elif (sys.argv[0] == "GR"):
    x = 0
    y = 0
    Generate(x, y, Sigma, f)
    Metropolis()
else:
    print("Incorrect input")