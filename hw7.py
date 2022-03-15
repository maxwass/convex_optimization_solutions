import math
import cvxpy as cp, numpy as np, matplotlib.pyplot as plt, scipy
from typing import List

from data.sep3way_data import load_sep3way_data

####### 3 way separation #######
def three_way_simultaneous_feasibility(x: np.array, y:np.array, z: np.array):
    # Three-way linear classification from Homework 7 Additional Exercises
    assert x.ndim == 2 and y.ndim == 2 and z.ndim ==2 and x.shape[0] == y.shape[0] == z.shape[0]
    n = x.shape[0]
    a1, b1 = cp.Variable(n, name='a1'), cp.Variable(1, name='b1')
    a2, b2 = cp.Variable(n, name='a2'), cp.Variable(1, name='b2')
    a3, b3 = cp.Variable(n, name='a3'), cp.Variable(1, name='b3')
    hs = cp.hstack
    constraints = []
    constraints.append(x.T @ a1 - b1 >= cp.max(hs([x.T @ a2 - b2, x.T @ a3 - b3])) + 1.0)
    constraints.append(y.T @ a2 - b2 >= cp.max(hs([y.T @ a1 - b1, y.T @ a3 - b3])) + 1.0)
    constraints.append(z.T @ a3 - b3 >= cp.max(hs([z.T @ a1 - b1, z.T @ a2 - b2])) + 1.0)
    constraints.append(a1 + a2 + a3 == 0)
    constraints.append(b1 + b2 + b3 == 0)
    #objective = cp.norm2(a1) + cp.norm2(a2) + cp.norm2(a3) + cp.norm2(b1) + cp.norm2(b2) + cp.norm2(b3)
    objective = 0#
    problem = cp.Problem(cp.Minimize(objective), constraints)

    problem.solve(solver='MOSEK')
    funcs = [(a1.value, b1.value), (a2.value, b2.value), (a3.value, b3.value)]
    return funcs


## Helper funcs for plotting 3 way separation ##
def apply_func_meshgrid(func, x1s, x2s):
    return func[0][0] * x1s + func[0][1] * x2s - func[1]


def list_of_maximizing_points(out1, out2, out3, xs, ys):
    n, m = out1.shape
    out1_x_max, out2_x_max, out3_x_max = [], [], []
    out1_y_max, out2_y_max, out3_y_max = [], [], []
    for i in range(n):
        for j in range(m):
            if out1[i][j] > np.maximum(out2[i][j], out3[i][j]):
                out1_x_max.append(xs[i, j])
                out1_y_max.append(ys[i, j])
            elif out2[i][j] > np.maximum(out1[i][j], out3[i][j]):
                out2_x_max.append(xs[i, j])
                out2_y_max.append(ys[i, j])
            else:
                out3_x_max.append(xs[i, j])
                out3_y_max.append(ys[i, j])
    out1_pts = np.stack([np.array(out1_x_max), np.array(out1_y_max)])
    out2_pts = np.stack([np.array(out2_x_max), np.array(out2_y_max)])
    out3_pts = np.stack([np.array(out3_x_max), np.array(out3_y_max)])

    return out1_pts, out2_pts, out3_pts


def plot_sep3way(data, funcs, data_lims):
    # construct grid points of space, feed through funcs, color points by their maximum with high transparency
    x1_lims, x2_lims = data_lims
    x1 = np.linspace(x1_lims[0], x1_lims[1], 200)
    x2 = np.linspace(x2_lims[0], x2_lims[1], 200)
    x1v, x2v = np.meshgrid(x1, x2)

    # apply each func to each point
    out1 = apply_func_meshgrid(funcs[0], x1v, x2v)
    out2 = apply_func_meshgrid(funcs[1], x1v, x2v)
    out3 = apply_func_meshgrid(funcs[2], x1v, x2v)

    o1, o2, o3 = list_of_maximizing_points(out1, out2, out3, x1v, x2v)

    # plot these points in that regions respective color with high transparancy...
    plt.scatter(data[0][0], data[0][1], color='red')
    plt.scatter(data[1][0], data[1][1], color='blue')
    plt.scatter(data[2][0], data[2][1], color='green')
    plt.scatter(o1[0], o1[1], color='red', alpha=.1, s=5)
    plt.scatter(o2[0], o2[1], color='blue', alpha=.1, s=5)
    plt.scatter(o3[0], o3[1], color='green', alpha=.1, s=5)
    #plt.set_xlim(xlims[0], xlims[0])
    plt.show()
## ^^ Helper funcs for plotting 3 way separation ^^ ##



####### Numerical Experiments: 9.30 and 9.31 #######
# gradient f = A(1-A'x)^-1 + 1'(2x*(1-x^2)^-1)
#    where the ith column of A is a_i, inverses and  * are elementwise
# Hessian f = - BB' - 2diag( (1+x^2)^-1 + 2x^2(1-x^2)^-2 )
#    where the ith column of B = (1-a_i'x)^-1 a_i

def f(A: np.array, x: np.array):
    return - np.log(1 - A @ x).sum() - np.log(1 - x ** 2).sum()


def grad_f(A: np.array, x: np.array):
    g = A.T @ ((1 - A@x) ** (-1)) + 2 * x * (1 - x ** 2) ** (-1)
    return g


def hess_f(A: np.array, x: np.array):
    # B is A, but with each row a_i scaled  by (1-a_i'x)^-1
    B = (A.T @ np.diag((1 - A @ x) ** (-1))).T

    # contribution from -1'@log(1-Ax)
    hess_1 = B.T @ B
    # contribution from -1'@log(1-x**2)
    hess_2 = 2*np.diag( (1 - x**2)**(-1) + 2*(x**2)*(1 - x**2)**(-2) )
    return hess_1 + hess_2


def newton_step_via_chol(hess_chol_lower, grad):
    y = np.linalg.solve(hess_chol_lower, -grad)
    hess_inv_grad = np.linalg.solve(hess_chol_lower.T, y)
    return hess_inv_grad


def in_domain(A: np.array, x: np.array):
    # 1 - A@x > 0 && 1 - x**2 > 0
    return all(1-A @ x > 0) and all(1 - x**2 > 0)


def backtrack(A: np.array, x: np.array, dir: np.array, grad: np.array,
              a: float, b: float):

    t = 1
    while True:
        # check if in domain
        if in_domain(A, x + t*dir) and f(A, x + t*dir) < f(A, x) + a*t*grad.T@dir:
            break
        t = b*t
    return t

def gradient_descent(n: int = 100, m: int = 200,
                     alpha: float = .01, beta: float = .05,
                     eps: float = 1e-3, seed: int = 50):
    # sample problem data
    np.random.seed(seed)
    A = np.random.randn(m, n)

    # zero is an initial feasible point
    objective_values, grad_norms, step_sizes = [], [], []
    x = np.zeros(n)
    grad = grad_f(A, x)
    while np.linalg.norm(grad, 2) > eps:
        obj, grad = f(A, x), grad_f(A, x)
        desc_dir = - grad
        t = backtrack(A, x, desc_dir, grad, alpha, beta)
        x = x + t*desc_dir

        # log for plotting
        objective_values.append(obj)
        grad_norms.append(np.linalg.norm(grad, 2))
        step_sizes.append(t)

        print(f'{len(objective_values)}: objective {obj:.5f}, grad_norm: {np.linalg.norm(grad, 2):.5f}, t: {t:.5f}')

    optimal_value = objective_values[-1]
    x = np.arange(0, len(step_sizes))
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(x, np.log10(np.array(objective_values) - optimal_value))
    axs[1].plot(x, np.log10(grad_norms))
    marker_style = dict(color='tab:blue', linestyle='none', marker='x',
                        markersize=1, markerfacecoloralt='tab:blue')
    axs[2].plot(x, np.log10(step_sizes), fillstyle='none', **marker_style)
    axs[0].set_title('log10 f(x_k) - p^*')
    axs[1].set_title('log10 grad 2 norm')
    axs[2].set_title('log10 step size')

    plt.xlabel('iterations')
    plt.suptitle(f"Gradient Descent on -1'log(1-Ax) - 1'log(1-x^2) via Backtracking (a: {alpha:.2f}, b: {beta:.2f})")



def newton_method(n: int = 100, m: int = 200,
                  alpha: float = .01, beta: float = .05,
                  eps: float = 1e-6, seed: int = 50,
                  hess_freq: int = 1,
                  hess_approx=None):
    # sample problem data
    np.random.seed(seed)
    A = np.random.randn(m, n)

    # zero is an initial feasible point
    x = np.zeros(n)
    objective_values, newton_decrements, step_sizes, flops = [], [], [], []
    i = 0
    while i==0 or (newton_decr**2)/2 > eps:
        obj, grad = f(A, x), grad_f(A, x)
        if i % hess_freq == 0:
            hess = hess_f(A, x)
            hess = np.diag(np.diag(hess)) if hess_approx == 'diag' else hess
            L = np.linalg.cholesky(hess)
            flops.append((n**3)/3 + n**2)
        else:
            flops.append(n**2)
        newton_step = newton_step_via_chol(L, grad)
        desc_dir = newton_step
        newton_decr = np.sqrt((-grad.T @ newton_step))
        t = backtrack(A, x, desc_dir, grad, alpha, beta)
        x = x + t * desc_dir

        # log for plotting
        objective_values.append(obj)
        newton_decrements.append(newton_decr)
        step_sizes.append(t)
        i += 1

        print(f'{len(objective_values)}: objective {obj:.5f}, nd: {newton_decr:.5f}, t: {t:.5f}')
        #print(f'\tcondition of hess: {np.linalg.cond(hess):.5f}')

    x = np.cumsum(flops) #np.arange(0, len(step_sizes))
    fig, axs = plt.subplots(3, 1, sharex=True)
    optimal_value = objective_values[-1]
    axs[0].plot(x, np.log10(np.array(objective_values) - optimal_value))
    axs[1].plot(x, np.log10(newton_decrements))
    axs[2].plot(x, step_sizes)
    axs[0].set_title('log10 f(x_k) - p^*')
    axs[1].set_title('log10 newton_decr^2/2')
    axs[2].set_title('raw step size')

    plt.xlabel('flops')#'iterations')
    plt.suptitle(f"Newtons Method on -1'log(1-Ax) - 1'log(1-x^2) via Backtracking (a: {alpha:.2f}, b: {beta:.2f})")

if __name__ == "__main__":

    # three way separation
    """
    X, Y, Z = load_sep3way_data()
    # funcs = one_vs_rest_classifier([X, Y, Z])
    funcs = three_way_simultaneous_feasibility(X, Y, Z)
    plot_sep3way([X, Y, Z], funcs=funcs, data_lims=[(-7, 7), (-7, 7)])  # np.arange(start=-7, stop=7, step=.01))
    """

    # 9.30
    gradient_descent(seed=50, n=100, m=200)

    # 9.31 (a)
    # order of magnitude speed up (to convergence, measured in number of flops) by evaluating
    # hessian every 15 steps!
    for hess_freq in [1, 15, 30]:
        newton_method(n=1000, m=2000, eps=1e-10, hess_freq=hess_freq, hess_approx='diag')
        print(f'see')

    # 9.31 (b)
    newton_method(n=100, m=200, eps=1e-10, hess_freq=1, hess_approx='diag')
