import numpy as np
from numpy.linalg import eigvals, det
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp

rng = np.random.default_rng(42)

Roff = 122.8e3
MT = 3333
Gamma0Dt = 4.73e-5
C = 1.25e-13
Cp = 100e-12

eps = 1e-12

Is = np.array([5, 10, 600, 1000, 2000]) * 1e-6
xs = np.linspace(eps, 1-eps, 1000)
palette = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
palette10 = [mpl.colors.to_hex(c) for c in plt.cm.viridis(np.linspace(0.05, 0.95, 10))]

def get_x_eq(x, I):
    return I*I*Roff*np.log(1/x) - Gamma0Dt * (1 + (MT-1)*x*x)

def zero_x_eq(I):
    return brentq(get_x_eq, eps, 1-eps, args=(I,))
    # return fsolve(x_eq, x0=np.array([0.5]), args=np.array([I]))[0]

def zeros_x_eq(Is):
    x = np.zeros_like(Is)
    for i in range(len(Is)):
        x[i] = zero_x_eq(Is[i])
    return x

def get_v_eq(x, I):
    k1 = 1 + (MT -1)*x*x
    return (I*Roff) / k1

def get_equilibrium(I):
    x_eq = zero_x_eq(I)
    v_eq = get_v_eq(x_eq, I)
    return x_eq, v_eq

def get_equilibriums(Is):
    x_eq = zeros_x_eq(Is)
    v_eq = np.zeros_like(x_eq)
    for i in range(len(Is)):
        v_eq[i] = get_v_eq(x_eq[i], Is[i])
    return x_eq, v_eq

def system(t, s, I):
    v, x = s
    x = np.clip(x, eps, 1-eps)
    k1 = (1 + (MT-1)*x*x)/Roff
    dv = (I - k1*v)/Cp
    dx = (k1*v*v - Gamma0Dt/np.log(1/x))/C
    return [dv, dx]

def trajectory(I, v0, x0, T=40e-6):
    sol = solve_ivp(system, [0, T], [v0, x0], args=(I,), method="Radau", rtol=1e-8, atol=1e-12, dense_output=True, max_step=1e-9)
    return sol


def random_ics(I, n=10, xmin=1e-9, xmax=0.9, vmin=0.0, vmax=1.2):
    x0 = 10**rng.uniform(np.log10(xmin), np.log10(xmax), n)
    v0 = rng.uniform(vmin, vmax, n)
    return v0, x0

def plot_random_trajectories(I, colors, n=10, T=40e-6):
    xs_, vs_ = get_equilibrium(I)
    v0s, x0s = random_ics(I, n)
    plt.figure()
    i = 0
    for v0, x0 in zip(v0s, x0s):
        sol = trajectory(I, v0, x0, T)
        plt.plot(sol.y[1], sol.y[0], color=colors[i], lw=1, alpha=0.7)
        plt.plot(x0, v0, ".", color=colors[i])
        i+=1
    plt.plot(xs_, vs_, "ko")
    plt.xscale("log")
    plt.xlabel("x"); plt.ylabel("v (V)")
    plt.title(f"I = {I*1e6:.0f} µA")
    plt.show()


def print_eq(x_eq, v_eq):
    print("Equilibrium (x e6, v")
    for i in range(len(x_eq)):
        print(f"({x_eq[i]*1e6:.6f}, {v_eq[i]:.6f})")
    print(" ")

def get_jacobian(v, x):
    df1dx = -2 * (MT-1)*x*v/(Cp * Roff)
    df1dv = -(1+(MT-1)*x*x) / (Cp * Roff)
    df2dx = (2*(MT-1)*x*v*v/Roff - Gamma0Dt/(x * np.log(1/x)**2))/C
    df2dv = 2*v * (1+(MT-1)*x*x) / (C*Roff)
    J = np.zeros((2, 2))
    J[0, 0] = df1dv
    J[0, 1] = df1dx
    J[1, 0] = df2dv
    J[1, 1] = df2dx
    return J

def linearized(x, v, x_eq, v_eq, J):
    x_tild = x - x_eq
    v_tild = v - v_eq
    X = np.array([v_tild, x_tild])
    return J@X

def stability(J, rtol=1e-12):
    lam = eigvals(J)
    re, im = lam.real, lam.imag
    tol = rtol * np.max(np.abs(lam))

    attract_focus, rep_focus, center = 0, 0, 0
    attract_node, rep_node, saddle = 0, 0, 0
    non_hyper = 0

    complexe = np.max(np.abs(im)) > tol
    hyperbolic = bool(np.all(np.abs(re) > tol))

    if complexe:
        attract_focus = (re[0] < -tol)
        rep_focus = (re[0] > tol)
        center = not (attract_focus or rep_focus)
    else:
        attract_node = (np.all(re < -tol))
        rep_node = (np.all(re > tol))
        saddle =  (re.min() < -tol and re.max() > tol)
        non_hyper = not (attract_node + rep_node + saddle)

    nature = "Attractive focus" * attract_focus + "Repulsive focus" * rep_focus + "Center" * center + "Attractive node" * attract_node + "Repulsive node" * rep_node + "Saddle" * saddle + "Non hyperbolic" * non_hyper
    print("valeurs propres: ", lam)
    print("hyperbolique: ", hyperbolic)
    print("nature: ", nature)
    print(" ")
    return lam, hyperbolic, nature



def plot_x_eq(Is):
    plt.figure()
    i=0
    for I in Is:
        g = get_x_eq(xs, I)
        zero = zero_x_eq(I)
        plt.plot(xs, g, label=f"I = {I*1e6:.2f}uA; zero = {zero:.6f}", color=palette[i])
        plt.axvline(zero, linestyle=':', color=palette[i])
        i+=1

    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.title("Zeros for x equilibrium")
    plt.axhline(0, color='red', linestyle=':')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(-0.1, 0.1)
    plt.show()


styles = {
    "Attractive node":  dict(marker="o", filled=True),
    "Attractive focus": dict(marker="^", filled=True),
    "Repulsive node":   dict(marker="o", filled=False),
    "Repulsive focus":  dict(marker="^", filled=False),
    "Saddle":           dict(marker="s", filled=False),
    "Center":           dict(marker="D", filled=False),
}

def plot_trajectories(trajs, Is, colors, ncols=3):
    n = len(Is)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.ravel()

    for ax, I, c in zip(axes, Is, colors):
        for sol in trajs[I]:
            ax.plot(sol.y[1], sol.y[0], color=c, lw=1, alpha=0.7)
            ax.plot(sol.y[1][0], sol.y[0][0], ".", color=c)

        x_s, v_s = get_equilibrium(I)
        _, _, nature = stability(get_jacobian(v_s, x_s), rtol=1e-12)
        st = styles.get(nature, dict(marker="x", filled=True))
        ax.scatter(x_s, v_s, s=80, zorder=3, marker=st["marker"],
                   facecolors=c if st["filled"] else "none",
                   edgecolors="k", label=nature)

        ax.set_xscale("log")
        ax.set_xlabel("x"); ax.set_ylabel("v (V)")
        ax.set_title(f"I = {I*1e6:.0f} µA")
        ax.legend(loc="best")
        # ax.grid()

    for ax in axes[n:]: ax.axis("off")
    fig.tight_layout()
    plt.show()

def comput_Im(v, x):
    return (1 + (MT-1)*x*x)/ Roff * v

def get_im(traj):
    v = traj.y[0]
    x = traj.y[1]
    n = len(x)
    im = np.zeros_like(v)
    for i in range(n): im[i] = comput_Im(v[i], x[i])
    return im
def plot_im(trajs, Is, colors, ncols=3):
    n = len(Is)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.ravel()

    for ax, I, c in zip(axes, Is, colors):
        for sol in trajs[I]:
            im = get_im(sol)
            ax.plot(sol.y[0], im, color=c, lw=1, alpha=0.7)
            ax.plot(sol.y[0][0], im[0], ".", color=c)

        ax.set_xscale("log")
        ax.set_xlabel("v (V)"); ax.set_ylabel(r"$i_M$ (A)")
        ax.set_title(f"I = {I*1e6:.0f} µA")
        ax.legend(loc="best")
    for ax in axes[n:]: ax.axis("off")
    fig.tight_layout()
    plt.show()

### Q2 ###
# plot_x_eq(Is)
x_eq, v_eq = get_equilibriums(Is)
# # print_eq(x_eq, v_eq)
# # plot_random_trajectories(Is[0], palette10)
# J0 = get_jacobian(v_eq[0], x_eq[0])
# J1 = get_jacobian(v_eq[1], x_eq[1])
# J2 = get_jacobian(v_eq[2], x_eq[2])
# J3 = get_jacobian(v_eq[3], x_eq[3])
# J4 = get_jacobian(v_eq[4], x_eq[4])
# stability(J0)
# stability(J1)
# stability(J2)
# stability(J3)
# stability(J4)

### Q3 ###
I0 = 5e-6
I1 = 20e-6
v0 = 0.5
x0 = 1e-6

traj0 = trajectory(I0, v0, x0, T=60e-6)
traj1 = trajectory(I1, v0, x0, T=10e-6) # Does the solution cycle ? Seems to when we increase the simulation time

trajs = {I0: [traj0], I1: [traj1]}
plot_trajectories(trajs, [I0, I1], [palette[0], palette[1]], ncols=2)
plot_im(trajs, [I0, I1], [palette[0], palette[1]], ncols=2)

