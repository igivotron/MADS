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

### Q2 ###
Is = np.array([5, 10, 600, 1000, 2000]) * 1e-6
xs = np.linspace(eps, 1-eps, 1000)
palette = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
palette10 = [mpl.colors.to_hex(c) for c in plt.cm.viridis(np.linspace(0.05, 0.95, 10))]

def get_x_eq(x, I):
    return I*I*Roff*np.log(1/x) - Gamma0Dt * (1 + (MT-1)*x*x)

def zero_x_eq(I):
    return brentq(get_x_eq, eps, 1-eps, args=np.array([I]))
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

def trajectory(I, v0, x0, T=1e-6):
    sol = solve_ivp(system, [0, T], [v0, x0], args=(I,), method="Radau", rtol=1e-8, atol=1e-12, dense_output=True)
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

def get_jacobian(x, v):
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
    X = np.array([x_tild, v_tild])
    return J@X

def stability(J):
    t = np.trace(J)
    d = det(J)
    l1, l2 = eigvals(J)
    print(l1, l2)

    hyperbolic = not (d == 0)
    attractive = (l1 < 0 and l2 < 0)
    repulsive = (l1 > 0 and l2 > 0)
    saddle = (min(l1, l2) < 0 and max(l1, l2)>0)
    attractive_eq = (max(l1, l2) == 0 and min(l1, l2) < 0)
    repulsive_eq = (max(l1, l2) > 0 and min(l1, l2) == 0)

    print(f"Hyperbolic: {hyperbolic}")
    print(f"Attractive: {attractive}")
    print(f"Repulsive: {repulsive}")
    print(f"Saddle: {saddle}")
    print(f"Attractive equilibrium: {attractive_eq}")
    print(f"Repulsive equilibrium: {repulsive_eq}")



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



# plot_x_eq(Is)
x_eq, v_eq = get_equilibriums(Is)
# print_eq(x_eq, v_eq)
J = get_jacobian(x_eq[0], v_eq[0])
stability(J)