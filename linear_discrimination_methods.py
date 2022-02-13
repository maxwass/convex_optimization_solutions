import cvxpy as cp, numpy as np, matplotlib.pyplot as plt
from typing import List


def one_vs_rest_classifier(data: List[np.array]):
    # data is a list of matrices - the columns of which are data points

    funcs = []
    for i in range(len(data)):
        x = data[i]
        rest_data = construct_rest_data(data, i)
        p = robust_linear_discrimination(x, rest_data)
        p_solve = p.solve(solver='MOSEK')
        funcs.append((p.var_dict['a'].value, p.var_dict['b'].value))
    return funcs


def construct_rest_data(data, idx_of_positive_class):
    # in one-vs-rest method, combine all the non-positive class data together
    rest_idxs = [i for i in range(len(data))]
    del rest_idxs[idx_of_positive_class]
    rest_data = np.concatenate([data[i] for i in rest_idxs], axis=1)
    return rest_data


def robust_linear_discrimination(X: np.array, Y: np.array):
    # columns of X/Y  are data point
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0]
    n = X.shape[0]
    a = cp.Variable(n, name='a')
    b = cp.Variable(1, name='b')
    t = cp.Variable(1, name='t')

    constraints = []
    constraints.append(X.T @ a - b >= t)
    constraints.append(Y.T @ a - b <= -t)
    constraints.append(cp.norm(a, p=2) <= 1)
    problem = cp.Problem(cp.Maximize(t), constraints)
    return problem


def linear_discrimination(X: np.array, Y: np.array):
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[0] == Y.shape[0]
    n = X.shape[0]
    a = cp.Variable(n, name='a')
    b = cp.Variable(1, name='b')
    constraints = []
    ## strict ineq's will result in error thrown
    #constraints.append(lower_left.T @ a1 - b1 > 0)
    #constraints.append(upper_right.T @ a1 - b1 < 0)
    ## weak inequalities allow trivial zero solution
    #constraints.append(lower_left.T @ a1 - b1 >= 0)
    #constraints.append(upper_right.T @ a1 - b1 <= 0)
    ## add margin - correct
    constraints.append(lower_left.T @ a - b >= 1)
    constraints.append(upper_right.T @ a - b <= -1)
    objective = 0
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver='MOSEK', verbose=True)



# lower left quadrant x1, x2 \in [-1, .25]
num_points, n = 20, 2
lower_left = np.random.uniform(low=-1, high=-.25, size=(n, num_points))
upper_right = np.random.uniform(low=.25, high=1, size=(n, num_points))
linear_discrimination(lower_left, upper_right)
