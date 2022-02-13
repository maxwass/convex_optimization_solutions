import cvxpy as cp, numpy as np, matplotlib.pyplot as plt
from typing import List

from data.sep3way_data import load_sep3way_data


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

X, Y, Z = load_sep3way_data()
#funcs = one_vs_rest_classifier([X, Y, Z])
funcs = three_way_simultaneous_feasibility(X, Y, Z)
plot_sep3way([X, Y, Z], funcs=funcs, data_lims=[(-7, 7), (-7, 7)]) #np.arange(start=-7, stop=7, step=.01))