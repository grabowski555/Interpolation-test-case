#%%
"""
Compare closed-form analytic trajectory vs Boris (with exact Ey=sin x).
Assumes Ex=Ez=0, B=0, so x(t)=v0 t + x0 and Ey(x)=A sin x.
"""

import numpy as np
import matplotlib.pyplot as plt

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

A        = np.float64(1.0)    # Ey amplitude in Ey=A*sin(x)
q_over_m = np.float64(1.0)    # q/m
v0       = np.float64(1.0)    # initial vx
x0       = np.float64(0)# initial x (phase)
y0       = np.float64(0.0)    # initial y
vy0      = np.float64(-1)    # initial vy 
T        = 4*np.pi             # total time
Nt       = 4000                # time steps 

#grid
t  = np.linspace(0.0, T, Nt, dtype=np.float64)
dt = np.float64(t[1] - t[0])

#analytic sol
def x_analytic(t):
    return v0*t + x0

def Ey_exact(x):
    return A * np.sin(x, dtype=np.float64)

def vy_analytic(t):
    return (vy0
            - (q_over_m*A/v0)*(np.cos(v0*t + x0) - np.cos(x0)))

def y_analytic(t):
    return (y0
            + (vy0 + (q_over_m*A/v0)*np.cos(x0))*t
            - (q_over_m*A/(v0**2))*(np.sin(v0*t + x0) - np.sin(x0)))
# def vy_analytic(t):
#     return (vy0
#             - (q_over_m*A/v0) * (np.cos(v0*t + x0) - np.cos(x0)))

# def y_analytic(t):
#     return (y0 + vy0*t
#             - (q_over_m*A/(v0**2)) * (np.sin(v0*t + x0) - np.sin(x0)))

def boris_step_core_exactE(x, vxh, vyh, dt):
    """One staggered Boris step with Ex=Ez=0, B=0, Ey=Ey_exact(x)."""
    Ex = np.float64(0.0)
    Ey = Ey_exact(x)

    vxm = vxh + 0.5*dt*q_over_m*Ex   
    vym = vyh + 0.5*dt*q_over_m*Ey

    vxh_new = vxm + 0.5*dt*q_over_m*Ex  
    vyh_new = vym + 0.5*dt*q_over_m*Ey

    x_new = x + vxh_new*dt
    y_inc = vyh_new*dt
    return x_new, y_inc, vxh_new, vyh_new, Ey

def run_boris_exact():
    x = np.float64(x0)
    y = np.float64(y0)

    Ey0 = Ey_exact(x0)
    vxh = np.float64(v0)                 
    vyh = np.float64(vy0) + 0.5*dt*q_over_m*Ey0

    xs = np.empty(Nt, dtype=np.float64)
    ys = np.empty(Nt, dtype=np.float64)
    Ey_series = np.empty(Nt, dtype=np.float64)
    vxh_series = np.empty(Nt, dtype=np.float64)
    vyh_series = np.empty(Nt, dtype=np.float64)

    for n in range(Nt):
        xs[n] = x
        ys[n] = y
        vxh_series[n] = vxh
        vyh_series[n] = vyh
        Ey_series[n] = Ey_exact(x) 

        if n < Nt-1:
            x, dy, vxh, vyh, _ = boris_step_core_exactE(x, vxh, vyh, dt)
            y = y + dy

    return xs, ys, Ey_series, vxh_series, vyh_series

x_b, y_b, Ey_b, vxh_b, vyh_b = run_boris_exact()

def KE_half(vxh, vyh):
    return 0.5*(vxh*vxh + vyh*vyh)

KE_b = KE_half(vxh_b, vyh_b)

KE_an = 0.5*(v0**2 + vy_analytic(t)**2)

rmse = lambda a,b: np.sqrt(np.mean((a-b)**2, dtype=np.float64))
print("dt =", float(dt))
print("RMSE[y (Boris) vs y_analytic]     = %.6e" % rmse(y_b, y_analytic(t)))
print("RMSE[Ey (Boris) vs Ey_analytic]   = %.6e" % rmse(Ey_b, Ey_exact(x_analytic(t))))
print("RMSE[KE (Boris) vs KE_analytic]   = %.6e" % rmse(KE_b, KE_an))

# y(t)
plt.figure(figsize=(9,4.2))
plt.plot(t, y_analytic(t), lw=2, label="analytic y(t)")
plt.plot(t, y_b, lw=1.4, ls="--", label="Boris + exact Ey")
plt.title("y(t): Analytic vs Boris ")
plt.xlabel("t"); plt.ylabel("y(t)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

# (2) Ey along the path
plt.figure(figsize=(9,4.2))
plt.plot(t, Ey_exact(x_analytic(t)), lw=2, label="analytic Ey(x(t))")
plt.plot(t, Ey_b, lw=1.4, ls="--", label="Boris Ey(x^n)")
plt.title("Field along the path: analytic vs Boris sampling")
plt.xlabel("t"); plt.ylabel("Ey")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

# (3) Kinetic energy 
plt.figure(figsize=(9,4.2))
plt.plot(t, KE_an, lw=2, label="analytic KE(t)")
plt.plot(t, KE_b,  lw=1.4, ls="--", label="Boris KE (half-step v)")
plt.title("Kinetic energy: Analytic vs Boris ")
plt.xlabel("t"); plt.ylabel("K(t) = 0.5 (vx^2 + vy^2)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()

plt.show()
# %%
