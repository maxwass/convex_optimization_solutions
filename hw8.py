import math
import cvxpy as cp, numpy as np, matplotlib.pyplot as plt, scipy
from typing import List

#### Part 1: Netwon method, centering ####
def sample_problem_data(m: int, n: int, seed: int):
    # sample problem data according to instructions

    np.random.seed(seed)

    A = np.random.randn(m - 1, n)
    # ensure sublevel set is bounded (if not, centering problem is unbounded below): add row with strictly
    #  positive elements
    A = np.vstack([A, np.random.rand(n) + 1e-4])
    rank_a = np.linalg.matrix_rank(A)
    cond_a = np.linalg.cond(A)
    assert 1e-3 < cond_a < 1e3, f'A poorly conditioned: {cond_a}'
    assert rank_a == m, f'A not full rank: rank(A) = {rank_a} < {m}'

    # x_0 >= 0, b \in Range(A) by construction -> x_0 feasible
    x_0 = np.random.rand(n)
    b = A @ x_0
    # c chosen randomly
    c = np.random.rand(n)

    return A, b, x_0, c


def f(c, x):
    assert all(x >= 0), f'x has negative entry, out of domain'
    return c @ x - np.log(x).sum()


def newton_step_block_elim(A: np.ndarray, x: np.ndarray, c: np.ndarray):
    # solve for newton step and optimal nu via block elimination of the diagonal hession (1x1 block)

    # form gradient, inverse hessian, schur complement
    g = c - 1/x
    h_inv = np.diag(x**2)
    s = A @ h_inv @ A.T

    # solve for nu^*
    nu = scipy.linalg.solve(a=s, b=- A @ h_inv @ g, assume_a='sym')

    # plug in for newton step delta_x
    delta_x = - h_inv @ (A.T @ nu + g)

    # compare to directly solving full KKT
    #KKT = np.block([h_])

    return delta_x, nu


def backtrack(x: np.ndarray, dir: np.ndarray, grad: np.ndarray, c:np.ndarray,
              alpha: float, beta: float):

    t = 1
    while True:
        # for a point to be a valid update point, we must first check if the proposed
        # point is feasible, which here means that it is:
        # i) non-neg.
        # ii) in null space of A. This is always true by construction, no need to check here.
        # only if these are true, do we then proceed to check if it satisfies backtracking condition.
        if (x + t*dir > 0).all() and f(c, x + t*dir) <= f(c, x) + alpha*t*grad.T@dir:
            break
        t = beta*t
    return t


def newton_method_centering(A: np.ndarray, c: np.ndarray, x_0: np.ndarray,
                            alpha: float = .1, beta: float = .5,
                            eps: float = 1e-6):
    # we assume x_0 is feasible

    objective_values, newton_decrements, step_sizes, flops = [], [], [], []
    newton_decr_sq, x = np.inf, x_0
    while newton_decr_sq > eps:
        obj, grad = f(c, x), c - 1/x
        newton_step, nu = newton_step_block_elim(A, x, c)
        newton_decr_sq = - (newton_step.T @ grad)/2  # recommended formula from assignment
        t = backtrack(x, newton_step, grad, c, alpha, beta)
        x = x + t * newton_step

        # log for plotting
        objective_values.append(obj)
        newton_decrements.append(newton_decr_sq)
        step_sizes.append(t)

        print(f'{len(objective_values)}: objective {obj:.5f}, nd: {newton_decr_sq:.5f}, t: {t:.5f}')
        #print(f'\tcondition of hess: {np.linalg.cond(hess):.5f}')

    info = {'objective_values': np.array(objective_values),
            'newton_decrements_sq': np.array(newton_decrements),
            'step_sizes': step_sizes,
            'num_steps': len(step_sizes)
            }

    return x, nu, info

#### Part 2: LP solver with strictly feasible starting point. ####

#### Part 3 ####


if __name__ == "__main__":

    # sample problem data
    A, b, x_0, c = sample_problem_data(m=500, n=2000, seed=50)
    alpha, beta = .1, .2
    x, nu, info = newton_method_centering(A, c, x_0, alpha=alpha, beta=beta, eps=1e-6)

    # do returned x, nu approximately solve the kkt conditions?
    # optimality: grad Lagrangian = 0?
    dual_residual = c - (1/x).sum() + A.T @ nu
    primal_residual = A @ x - b
    print(f'||dual residual||_2: {np.linalg.norm(dual_residual):.6f},||primal residual||_2: {np.linalg.norm(primal_residual):.6f}')

    xs = np.arange(0, len(info['step_sizes'])) #np.cumsum(flops) #
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(xs, np.log10(info['objective_values'] - info['objective_values'][-1] + 1e-16))
    axs[1].plot(xs, np.log10(info['newton_decrements_sq']))
    axs[2].plot(xs, info['step_sizes'])
    axs[0].set_title('log10 f(x_k) - p^*')
    axs[1].set_title('log10 newton_decr^2/2')
    axs[2].set_title('raw step size')
    plt.xlabel('iterations') #'flops')
    plt.suptitle(f"Newtons Method on c.T @ x - 1 @ log(x) s.t. Ax=b, via Backtracking (a: {alpha:.2f}, b: {beta:.2f})")
    plt.show()
    print('hi')

    # barrier method for LP