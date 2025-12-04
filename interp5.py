#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.family": "Bell MT",    # Font name
    "font.size":10,             # Global font size in points
    "axes.titlesize": 18,        # Title size
    "axes.labelsize": 18,        # X/Y label size
    "xtick.labelsize": 13,       # Tick label size
    "ytick.labelsize": 13,
    "legend.fontsize": 10,       # Legend text size
    "figure.titlesize": 20,      # Suptitle 
})

# grid
Lx_min, Lx_max = -2*np.pi, 2*np.pi
Ly_min, Ly_max = 0.0, 2*np.pi
Lx = np.float64(Lx_max - Lx_min)
Ly = np.float64(Ly_max - Ly_min)

Nx, Ny = 64, 64
xg = np.linspace(Lx_min, Lx_max, Nx, endpoint=False, dtype=np.float64)
yg = np.linspace(Ly_min, Ly_max, Ny, endpoint=False, dtype=np.float64)
dx = np.float64(Lx / Nx)
dy = np.float64(Ly / Ny)

def wrap(z, zmin, L):
    return (z - zmin) % L + zmin

def Ey_clean(x):
    return np.sin(x, dtype=np.float64)

#build noisy fields
Ey_grid_clean = Ey_clean(xg)[:, None] * np.ones((1, Ny), dtype=np.float64)

noise_on   = True
noise_amp  = np.float64(0.10)        
noise_mode = "absolute"              # "absolute" or "relative"
rng        = np.random.default_rng(42)

def sample_noise(scale=np.float64(1.0)):
    return scale * (np.float64(2.0)*rng.random() - np.float64(1.0))

def add_noise_to_grid(E_grid):
    noisy = E_grid.copy()
    for i in range(Nx):
        for j in range(Ny):
            if noise_mode == "absolute":
                noisy[i, j] += sample_noise(noise_amp)
            elif noise_mode == "relative":
                noisy[i, j] += sample_noise(noise_amp * (np.abs(E_grid[i, j]) + 1.0e-12))
            else:
                raise ValueError("noise_mode must be 'absolute' or 'relative'")
    return noisy

Ey_grid = add_noise_to_grid(Ey_grid_clean) if noise_on else Ey_grid_clean.copy()
deltaEy_grid = Ey_grid - Ey_grid_clean  # <-- perturbations only

T, Nt = 4*np.pi, 4000
t = np.linspace(0.0, T, Nt, dtype=np.float64)
v0, x0, y0 = np.float64(1.0), np.float64(0.0), np.float64(0.0)
x_path = wrap(v0*t + x0, Lx_min, Lx)
y_path = np.full_like(x_path, y0)

def dEy_nn(x, y):
    xr = wrap(x, Lx_min, Lx); yr = wrap(y, Ly_min, Ly)
    ix = np.rint((xr - Lx_min)/dx).astype(np.int64) % Nx
    iy = np.rint((yr - Ly_min)/dy).astype(np.int64) % Ny
    return deltaEy_grid[ix, iy]

def dEy_bilinear(x, y):
    xr = wrap(x, Lx_min, Lx); yr = wrap(y, Ly_min, Ly)
    fx = (xr - Lx_min)/dx;     fy = (yr - Ly_min)/dy
    i0 = np.floor(fx).astype(np.int64) % Nx
    j0 = np.floor(fy).astype(np.int64) % Ny
    i1 = (i0 + 1) % Nx;        j1 = (j0 + 1) % Ny
    ax = (fx - np.floor(fx)).astype(np.float64)
    ay = (fy - np.floor(fy)).astype(np.float64)
    w00 = (1.0 - ax)*(1.0 - ay); w10 = ax*(1.0 - ay)
    w01 = (1.0 - ax)*ay;         w11 = ax*ay
    return (w00*deltaEy_grid[i0, j0] + w10*deltaEy_grid[i1, j0] +
            w01*deltaEy_grid[i0, j1] + w11*deltaEy_grid[i1, j1])

def cubic_kernel(r, a=-0.5):
    r = np.abs(r)
    out = np.zeros_like(r, dtype=np.float64)
    m1 = (r < 1.0); m2 = (r >= 1.0) & (r < 2.0)
    out[m1] = ((a + 2.0)*r[m1] - (a + 3.0)) * r[m1]*r[m1] + 1.0
    out[m2] = ((a*r[m2] - 5.0*a)*r[m2] + 8.0*a)*r[m2] - 4.0*a
    return out

def dEy_bicubic_kernel(x, y, a=-0.5):
    xr = wrap(x, Lx_min, Lx); yr = wrap(y, Ly_min, Ly)
    fx = (xr - Lx_min)/dx;     fy = (yr - Ly_min)/dy
    i0 = np.floor(fx).astype(np.int64); j0 = np.floor(fy).astype(np.int64)
    ixm1 = (i0 - 1) % Nx; ix0 = i0 % Nx; ix1 = (i0 + 1) % Nx; ix2 = (i0 + 2) % Nx
    jym1 = (j0 - 1) % Ny; jy0 = j0 % Ny; jy1 = (j0 + 1) % Ny; jy2 = (j0 + 2) % Ny
    u = (fx - np.floor(fx)).astype(np.float64); v = (fy - np.floor(fy)).astype(np.float64)

    wx = np.stack([cubic_kernel(u + 1.0, a),
                   cubic_kernel(u + 0.0, a),
                   cubic_kernel(1.0 - u, a),
                   cubic_kernel(2.0 - u, a)], axis=0)
    wy = np.stack([cubic_kernel(v + 1.0, a),
                   cubic_kernel(v + 0.0, a),
                   cubic_kernel(1.0 - v, a),
                   cubic_kernel(2.0 - v, a)], axis=0)
    S_m1 = wx[0]*deltaEy_grid[ixm1, jym1] + wx[1]*deltaEy_grid[ix0, jym1] + wx[2]*deltaEy_grid[ix1, jym1] + wx[3]*deltaEy_grid[ix2, jym1]
    S_0  = wx[0]*deltaEy_grid[ixm1, jy0 ] + wx[1]*deltaEy_grid[ix0, jy0 ] + wx[2]*deltaEy_grid[ix1, jy0 ] + wx[3]*deltaEy_grid[ix2, jy0 ]
    S_1  = wx[0]*deltaEy_grid[ixm1, jy1 ] + wx[1]*deltaEy_grid[ix0, jy1 ] + wx[2]*deltaEy_grid[ix1, jy1 ] + wx[3]*deltaEy_grid[ix2, jy1 ]
    S_2  = wx[0]*deltaEy_grid[ixm1, jy2 ] + wx[1]*deltaEy_grid[ix0, jy2 ] + wx[2]*deltaEy_grid[ix1, jy2 ] + wx[3]*deltaEy_grid[ix2, jy2 ]
    return (wy[0]*S_m1 + wy[1]*S_0 + wy[2]*S_1 + wy[3]*S_2)


row_for_spline = 0  
xg_spline = np.linspace(Lx_min, Lx_max, Nx + 1, endpoint=True, dtype=np.float64)
dEy_samples = np.append(deltaEy_grid[:, row_for_spline], deltaEy_grid[0, row_for_spline])
dEy_samples[-1] = dEy_samples[0]  
cs_dEy = CubicSpline(xg_spline, dEy_samples, bc_type='periodic')

def dEy_cubic_periodic(x, y):
    xr = wrap(x, Lx_min, Lx)
    return np.float64(cs_dEy(xr))

dEy_true = np.zeros_like(t, dtype=np.float64)  # analytic perturbation = 0
dEy_nn = dEy_nn(x_path, y_path)
dEy_li = dEy_bilinear(x_path, y_path)
dEy_cu = dEy_bicubic_kernel(x_path, y_path)
dEy_cp = dEy_cubic_periodic(x_path, y_path)

#plots
rmse = lambda a: np.sqrt(np.mean(a*a, dtype=np.float64))
print("---- Settings ----")
print(f"Nx={Nx}, Ny={Ny}, noise_on={noise_on}, mode={noise_mode}, amp={float(noise_amp)}")
print(f"Path: v0={float(v0):.3f}, x0={float(x0):.3f}, y0={float(y0):.3f}, T={float(T):.3f}, Nt={Nt}")
print("\nRMSE of δEy(t) vs 0 (noise that passes through each interpolator):")
print("Nearest         : %.6e" % rmse(dEy_nn))
print("Bilinear        : %.6e" % rmse(dEy_li))
print("Bicubic kernel  : %.6e" % rmse(dEy_cu))
print("Cubic periodic  : %.6e" % rmse(dEy_cp))

plt.figure(figsize=(10,4.2))
plt.plot(t, dEy_nn, lw=1.1, label='nearest')
plt.plot(t, dEy_li, lw=1.1, label='bilinear')
plt.plot(t, dEy_cu, lw=1.1, label='bicubic kernel (a=-0.5)')
plt.plot(t, dEy_cp, lw=1.4, label='cubic periodic ')
plt.title(r"Perturbations along path: $\delta E_y(t)$")
plt.xlabel("t"); plt.ylabel(r"$\delta E_y(t)$")
plt.legend(ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.show()

# %%
