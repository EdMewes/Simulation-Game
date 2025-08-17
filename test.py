import numpy as np
import matplotlib.pyplot as plt
import src.physics_sim.heat_equation_solver as hes
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
G = gaussian2d((100, 100), sigma=12.0, amplitude=1.0, normalize=True)


sol = hes.CNM_solve(G)
for time in range(1):
    sol = hes.CNM_solve(sol)
    # print(sol.max())


# Visualize
plt.imshow(G, origin='lower')
plt.colorbar(label='Intensity')
plt.title('Centered 2D Gaussian (100×100, σ=12)')
plt.show()

plt.imshow(np.reshape(sol,(100,100)), origin='lower')
plt.colorbar(label='Intensity')
plt.title('Centered 2D Gaussian (100×100, σ=12)')
plt.show()