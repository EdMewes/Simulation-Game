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

        

    NOTES ON METHODS;
        - Diffuesion - implicit
        - Wave eq - explicit

"""

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.sparse import diags, csc_array
from scipy.sparse.linalg import spsolve
from physics_sim.timing import time_funtion as tf
import timeit

import dedalus.public as d3

import logging



from sympy.solvers.pde import pdsolve
from sympy import Function, Eq
from sympy.abc import x, y


logger = logging.getLogger(__name__)


test_size = 100
run_cases = 2
# @tf


def gaussian2d(shape=(100, 100), sigma=10.0, amplitude=1.0, normalize=False):
    h, w = shape
    y = np.arange(h) - (h - 1) / 2.0
    x = np.arange(w) - (w - 1) / 2.0
    X, Y = np.meshgrid(x, y)  # X varies along columns, Y along rows

    g = amplitude * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    if normalize:
        s = g.sum()
        if s != 0:
            g = g / s
    # print(np.shape(g))
    return g


# Example: 100x100, centered, sigma=12, normalized to sum to 1
data = gaussian2d((100, 100), sigma=12.0, amplitude=1.0, normalize=True)

# Banded matrix unusable at larger (>100x100 grid size due to memory requirements)
def CNM_band_matrix_build(x_size, y_size):
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

def CNM_band_solve(initial_data, x_size, band_mat):
    band_mat_next_step = solve_banded((x_size, x_size), band_mat, initial_data.ravel())
    return band_mat_next_step




def CNM_spmatrix_build(x_size, y_size):
    # Define size of simulation
    def_coef = 100
    delta_t = 0.05
    delta_x = 1

    dx2 = delta_x**2

    alpha = (delta_t * def_coef) / (2 * dx2)

    # fill values +- x
    outer_sub_diag = -alpha * np.ones(x_size * y_size - x_size)
    inner_sub_diag = -alpha * np.ones(x_size * y_size - 1)
    inner_sub_diag[x_size - 1 :: x_size] = 0
    main_diag = (1 + 4 * alpha) * np.ones(x_size * y_size)

    # initialise banded matrix
    sparse_band_mat = diags(
        [outer_sub_diag, inner_sub_diag, main_diag, inner_sub_diag, outer_sub_diag],
        [x_size, 1, 0, -1, -x_size],
        format="csc",
    )
    print("done")
    return sparse_band_mat


def CNM_spsolve(sp_matrix, initial_data):
    # Solve sparse matrix
    print("starting")
    band_mat_next_timestep = spsolve(sp_matrix, initial_data.ravel())
    return band_mat_next_timestep


# Sympy cannot solve 2D PDE, but can use for notation
# def sympy_build_eq():
#         return
# def sympy_sovle():
#     return


def scipy_sovle():
    return


def matrix_free_sovle():
    return


def FEniCS_sovle():
    return


def firedrake_sovle():
    return

# Fourier spectral methods - Periodic boundary, diffusion spills over
def dedalus_sovle_fourier():

    # Params
    timestepper = d3.SBDF2
    timestep = 0.05
    stop_sim_time = 20
    Nx = 100
    Ny = 100

    # Bases
    coords = d3.CartesianCoordinates('x', 'y') # maybe 'z'
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, 1))
    ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(0, 1))

    # Fields
    h = dist.Field(name='h',bases=(xbasis,ybasis))
    
    # Forcing
    f = dist.Field(name='f',bases=(xbasis,ybasis))
    f.fill_random('g', seed=40)    
    
    # Subs
    x, y = dist.local_grids(xbasis, ybasis)
    D = 1

    # Problem
    problem = d3.IVP([h], namespace=locals())
    problem.add_equation("dt(h) - D*lap(h) = f")

    # Solve
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)
    snapshots.add_task(h, name='temp')

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

    return 0


if __name__ == "__main__":
    spmat = CNM_spmatrix_build(test_size, test_size)
    data = np.random.rand(test_size, test_size)
    time = timeit.timeit(lambda: CNM_spsolve(spmat, data), number=run_cases)
    print("Average runtime of sparse:", time / run_cases)
