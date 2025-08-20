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
from scipy.sparse import diags, csc_array
from scipy.sparse.linalg import spsolve
from physics_sim.timing import time_funtion as tf
import timeit


# @tf

def CNM_matrix_build(x_size, y_size):
    # Define size of simulation
    def_coef = 100
    delta_t = 0.05
    delta_x = 1

    dx2 = delta_x**2

    alpha = (delta_t * def_coef) / (2 * dx2)

    # initialise banded matrix
    band_mat = np.zeros((2 * x_size + 1, x_size * y_size))

    # fill banded matrix
    # fill values +-y, edge cases accounted for
    band_mat[0, x_size:] = -alpha
    band_mat[2 * x_size, :-x_size] = -alpha

    # FASTER
    row = band_mat[x_size - 1]
    row.fill(-alpha)
    row[::x_size] = 0
    band_mat[x_size, :] = 1 + 4 * alpha
    row = band_mat[x_size + 1]
    row.fill(-alpha)
    row[::x_size] = 0

    return band_mat


def CNM_solve(initial_data, x_size, band_mat):
    

    # print(np.shape(band_mat))
    # print(band_mat)
    band_mat_next_step = solve_banded(
        (x_size, x_size), band_mat, initial_data.ravel()
    )
    return band_mat_next_step


bmat = CNM_matrix_build(100,100)

data = np.random.rand(100, 100)
# time = timeit.timeit(lambda: CNM_spmatrix_build(100, 100, data), number=10)
time = timeit.timeit(lambda: CNM_solve(data, 100, bmat), number=100)
print("Average runtime of banded:", time/100)








def CNM_spmatrix_build(x_size, y_size):
    # Define size of simulation
    def_coef = 100
    delta_t = 0.05
    delta_x = 1

    dx2 = delta_x**2

    alpha = (delta_t * def_coef) / (2 * dx2)

    # fill values +- x, edge cases accounted for
    outer_sub_diag = -alpha*np.ones(x_size*y_size-x_size)
    inner_sub_diag = -alpha*np.ones(x_size*y_size-1)
    inner_sub_diag[x_size-1::x_size] = 0
    main_diag = (1+4*alpha)*np.ones(x_size*y_size)

    # initialise banded matrix
    sparse_band_mat = diags([outer_sub_diag,inner_sub_diag,
                             main_diag, inner_sub_diag, outer_sub_diag],
                             [x_size, 1, 0, -1, -x_size], format='csc')

    return sparse_band_mat

def CNM_spsolve(sp_matrix, initial_data):
    # Solve sparse matrix
    band_mat_next_timestep = spsolve(sp_matrix, initial_data.ravel())
    return band_mat_next_timestep

spmat = CNM_spmatrix_build(100,100)

data = np.random.rand(100, 100)
# time = timeit.timeit(lambda: CNM_spmatrix_build(100, 100, data), number=10)
time = timeit.timeit(lambda: CNM_spsolve(spmat, data), number=100)
print("Average runtime of sparse:", time/100)







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
