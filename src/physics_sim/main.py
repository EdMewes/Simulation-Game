"""
Dedalus script solving the 2D Poisson equation with mixed boundary conditions.
This script demonstrates solving a 2D Cartesian linear boundary value problem
and produces a plot of the solution. It should take just a few seconds to run.

We use a Fourier(x) * Chebyshev(y) discretization to solve the LBVP:
    dx(dx(u)) + dy(dy(u)) = f
    u(y=0) = g
    dy(u)(y=Ly) = h

For a scalar Laplacian on a finite interval, we need two tau terms. Here we
choose to lift them to the natural output (second derivative) basis.

To run and plot:
    $ python3 poisson.py
"""

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
from physics_sim import heat_equation_solver as het

def main():

    print(type(np.zeros((1,2))))

    het.dedalus_sovle_fourier()
    # het.dedalus_sovle_chebyshev()

    # print('hi there')


    return 0


hi = main()



