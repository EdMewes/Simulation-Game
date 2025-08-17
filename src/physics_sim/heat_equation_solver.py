"""
Equation:
    dh/dt = a * Laplacian h + source

    Use finite difference, scipy.linalg.
    solve_banded, 



    I will attemp miltiple ways to solve the
    heat diffusion equation:
        - CNM using a wide banded matrix
            (may attempt to reduce the band width)
        - Sympy
        - Scipy
        - Matrix-free
        - FEniCS
        - Firedrake
        - Dedalus

"""
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded

def diffusuion_eq():
    # Define size of simulation
    x_size = 100
    y_size = 100


    
    return