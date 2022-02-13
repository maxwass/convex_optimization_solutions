import cvxpy as cp, numpy as np, matplotlib.pyplot as plt

##### problem 4 #####
# problem data: each row is return
# Pj = P[j] => vector of asset returns for outcome pi_j
outcomes, assets = 10, 5
P = np.array([[3.5000, 1.1100, 1.1100, 1.0400, 1.0100],
              [0.5000, 0.9700, 0.9800, 1.0500, 1.0100],
              [0.5000, 0.9900, 0.9900, 0.9900, 1.0100],
              [0.5000, 1.0500, 1.0600, 0.9900, 1.0100],
              [0.5000, 1.1600, 0.9900, 1.0700, 1.0100],
              [0.5000, 0.9900, 0.9900, 1.0600, 1.0100],
              [0.5000, 0.9200, 1.0800, 0.9900, 1.0100],
              [0.5000, 1.1300, 1.1000, 0.9900, 1.0100],
              [0.5000, 0.9300, 0.9500, 1.0400, 1.0100],
              [3.5000, 0.9900, 0.9700, 0.9800, 1.0100]])
# pi = distribution on possible outcome => uniform
pi = np.ones(outcomes)/outcomes


def construct_problem(outcome_returns: np.array, outcome_distrib: np.array):
    outcomes, assets = outcome_returns.shape
    p = outcome_returns #cp.Parameter(value=outcome_returns, nonneg=True, name='P')
    pi = outcome_distrib # cp.Parameter(value=outcome_distrib, nonneg=True, name='pi')
    allocation = cp.Variable(assets, nonneg=True, name='allocation')
    returns = cp.Variable(outcomes, name='returns')
    objective = - pi @ cp.log(returns)
    constraints = [returns == p @ allocation, allocation >= 0, cp.sum(allocation) == 1]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem

def run_problem_4():
    p = construct_problem(P, pi)
    optimal_long_term_growth_rate = -1 * p.solve()
    optimal_asset_distribution = p.variables()[1].value
    optimal_returns = p.variables()[0].value

    # naive investment strategy growth rate
    naive_allocation = np.ones(assets)/assets
    naive_returns = P @ naive_allocation
    naive_gr = np.mean(np.log(naive_returns))


    print(f'optimal growth rate: {optimal_long_term_growth_rate: .4f}, naive growth rate: {naive_gr:.4f}')


#### problem 6: Heuristic suboptimal solution for Boolean LP ####
# create problem data: interpret x_i as decision of whether or not to accept job i,
# and -c_i as the (positive) revenue generated if i-th job completed. Ax <= b can be interpreted
# as limits on our m resources. A_ij is the amount of resource i


def relaxed_blp(resource_usage: np.array, resource_contraints: np.array, revenue: np.array):
    m, n = resource_usage.shape
    x = cp.Variable(n, name='x')
    objective = revenue.T @ x
    constraints = [resource_usage @ x <= resource_contraints, np.zeros(n) <= x, x <= np.ones(n)]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return problem

def run_problem_6():
    np.random.seed(0)
    n, m = 100, 300
    A = np.random.rand(m, n)
    b = A @ np.ones(n)/2 # ensures that an all ones solutions will not work (in BLP, only discrete x values)
    c = -np.random.rand(n, 1)

    p_rblp = relaxed_blp(resource_usage=A, resource_contraints=b,revenue=c)
    L = p_rblp.solve()
    x = p_rblp.variables()[0].value

    ts = np.linspace(0, 1, 100)
    thresholded_x = np.array([x > t for t in ts])
    objective_tx = np.array([np.squeeze(c.T @ tx) for tx in thresholded_x])
    max_constraint_violation = np.array([np.max(A @ tx - b) for tx in thresholded_x])
    feasible_tx = np.array([mcv <= 0 for mcv in max_constraint_violation])

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    # plot infeasible first
    axs[1].plot(ts[~feasible_tx], objective_tx[~feasible_tx], 'r--', label='infeasible')
    axs[1].plot(ts[feasible_tx], objective_tx[feasible_tx], 'g--', label='feasible')
    axs[0].plot(ts[~feasible_tx], max_constraint_violation[~feasible_tx], 'r--', label='infeasible')
    axs[0].plot(ts[feasible_tx], max_constraint_violation[feasible_tx], 'g--', label='feasible')
    axs[1].set_xlabel('Threshold Value')
    axs[1].set_ylabel('Objective')
    axs[0].set_ylabel('Max Constr Violation')
    axs[0].legend()
    plt.show()

    # find t which minimizes objetive value and gives feasible point
    a = list(zip(ts, list(objective_tx), max_constraint_violation))
    b = list(filter(lambda e: e[2]<0, a)) # remove infeasible points
    c = list(sorted(b, key=lambda e: e[1])) # extract objective
    U = c[0][1]
    print(f'Relaxed BLP has objective value L = {L:.4f}. Best threshold found is {c[0][0]:.3f} which produced value U = {c[0][1]:.4f}')
    print(f'\tThus x_hat = x*^rlx > {c[0][0]:.3f} is feasible, and <= U-L = {U-L:.4f} suboptimal')