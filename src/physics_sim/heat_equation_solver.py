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




def CNM_solve(initial_data):
    # Define size of simulation
    x_size = 100
    y_size = 100

    def_coef = 100
    delta_t = 1
    delta_x = 1

    dx2 = delta_x ** 2

    alpha = (delta_t * def_coef) / (2 * dx2) 

    # initialise banded matrix
    band_mat = np.zeros((2*x_size+1, x_size*y_size))

    # fill banded matrix
    # fill values +-y, edge cases accounted for
    band_mat[0, x_size:] = -alpha
    band_mat[2*x_size, :-x_size] = -alpha
    
    # fill values +- x, edge cases accounted for
    band_mat[x_size-1, 1:] = -alpha
    band_mat[x_size-1, ::x_size] = 0
    band_mat[x_size, :] = 1 + 4*alpha
    band_mat[x_size+1, :-1] = -alpha
    band_mat[x_size+1, :: -x_size] = 0
    # print(np.shape(band_mat), np.shape(initial_data.flatten()))
    band_mat_next_step = solve_banded((x_size, x_size), band_mat, initial_data.flatten())

    return band_mat_next_step

# print(CNM_solve())






def sympy_sovle():
    return


def scipy_sovle():
    return


def matrix_free_sovle():
    return


def FEniCS_sovle():
    return


def firedrake_sovle():
    return

def dedalus_sovle():
    return
