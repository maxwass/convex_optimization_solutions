import cvxpy as cp, numpy as np, matplotlib.pyplot as plt

### problem 1 ###
def construct_problem(t_i, y_i):
    # reformulate into LP
    a = cp.Variable(3, name='a')
    b = cp.Variable(2, name='b')
    g = cp.Parameter(pos=True, name='g')

    constraints = []
    for t, y in zip(t_i, y_i):
        num = a.T @ np.array([1, t, t ** 2])
        denom = 1 + b.T @ np.array([t, t ** 2])
        diff = num - y*denom
        constraints.append(diff <= g*denom)
        constraints.append(-g*denom <= diff)
        #constraints.append(denom >= 0) # not needed, denom cannot be neg by above constraints
    # giving an objective of 0 makes it a feasbility problem
    problem = cp.Problem(cp.Minimize(0), constraints)
    return problem, g


def bisect(t_i, y_i, low=1e-4, high=0.5, tol=1e-4):
    # bisection to find optimal g
    l, h = low, high
    lp_problem, g = construct_problem(t_i, y_i)
    while (h-l) >= tol:
        g.value = (l + h) / 2 # this changes the Parameter objects value -> much faster to resolve
        lp_problem.solve()
        if lp_problem.status == 'infeasible':
            print(f'{g.value:.5f} -> {lp_problem.status}. Loosen constraint.')
            l = g.value
        else:
            print(f'{g.value:.5f} -> {lp_problem.status}. Tighten constraint.')
            h = g.value
        print(f'\tbisection window: [{l:.3f}, {h:.3f}] -> width {h-l:.4f}')

    a_hat, b_hat = lp_problem.var_dict['a'].value, lp_problem.var_dict['b'].value
    print(f"optimal value of a {a_hat}, b {b_hat}, g {g.value}")
    return a_hat, b_hat


def f(a, b, t):
    return a.T @ np.array([1, t, t ** 2]) / (1 + b.T @ np.array([t, t ** 2]))


def run_problem_1():
    # problem data
    k = 201
    i = np.arange(1, k + 1)
    t_i = -3 + 6 * (i - 1) / (k - 1)
    y_i = np.exp(t_i)

    a_hat, b_hat = bisect(t_i, y_i)
    pred = np.array([f(a_hat, b_hat, t) for t in t_i])

    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(t_i, y_i, 'ro', t_i, pred)
    ax[0].set_title('function fit')
    ax[1].plot(t_i, pred-y_i) #, t_i, np.abs(pred-y_i))
    ax[1].set_title('error')
    plt.show()

### problem 4: Robust least-squares with interval coefficient matrix ###
def robust_problem(A_bar, b, R):
    m, n = A_bar.shape
    y, x, z = cp.Variable(m, name='y'), cp.Variable(n, name='x'), cp.Variable(n, name='z')
    objective = cp.square(cp.norm(y, p=2))
    constraints = [-1*z <= x, x <= z,
                   A_bar @ x + R @ z - b <= y,
                   -1*y <= A_bar @ x - R @ z - b]
    return cp.Problem(cp.Minimize(objective), constraints)


def ls_problem(A, b):
    m, n = A.shape
    x = cp.Variable(n, name='x')
    objective = cp.square(cp.norm(A@x - b, p=2))
    return cp.Problem(cp.Minimize(objective))


def worst_case_residual_norm(x_optimal, A_bar, b, R):
    # now x is fixed, A is variable -> can directly encode variability in A
    m, n = A_bar.shape
    y, z = cp.Variable(m, name='z'), cp.Variable(n, name='z')
    #z = np.abs(x_optimal)
    objective = cp.square(cp.norm(y, p=2))
    constraints = [-1*z <= x_optimal, x_optimal <= z,
                   A_bar @ x_optimal + R @ z - b <= y,
                   -1*y <= A_bar @ x_optimal - R @ z - b]
    return cp.Problem(cp.Minimize(objective), constraints)


def run_problem_4():
    A_bar = np.array([[60.0, 45, -8],
                      [90.0, 30, -30],
                      [0.00, -8, -4],
                      [30.0, 10, -10]])
    b = np.array([-6, -3, 18, -9]).T
    R = np.ones_like(A_bar) * .05

    ls_prob = ls_problem(A_bar, b)
    p_ls = ls_prob.solve(solver='MOSEK')
    x_ls = ls_prob.var_dict['x'].value
    print(f"Non-robust residual norm and soln {p_ls:.6f}, {x_ls}")

    rob_problem = robust_problem(A_bar, b, R)
    p_rob = rob_problem.solve(solver='MOSEK')
    x_rob = rob_problem.var_dict['x'].value
    print(f"robust residual norm and soln {p_rob:.6f}, {x_rob}")

    # nominal residual norm should be same as objective value for ls -> this is what was optimized
    # robust should not go up in value!
    ls_nom_rs = np.linalg.norm(A_bar @ x_ls - b, ord=2)**2
    rob_norm_rs = np.linalg.norm(A_bar @ x_rob - b, ord=2)**2
    print(f"Nominal (A_bar) residual norm: ls {ls_nom_rs:.4f}, rob {rob_norm_rs:.4f}")

    # worst case residual norm should be same as objective value for rob -> this is what was optimized
    e_wc_rob = worst_case_residual_norm(x_rob, A_bar, b, R).solve(solver='MOSEK')
    e_wc_ls = worst_case_residual_norm(x_ls, A_bar, b, R).solve(solver='MOSEK')
    print(f"Worst Case (A_bar) residual norm: ls {e_wc_ls:.4f}, rob {e_wc_rob:.4f}")


### problem 5: interpolation of image ###
def image_interpolation(U_orig, known_idxs, p=2):
    m, n = U_orig.shape
    U = cp.Variable((m, n), name='U')
    f_vert, f_horz = construct_f(m), construct_f(n)
    vertical_diffs, horizontal_diffs = cp.vec(f_vert @ U), cp.vec(f_horz @ U.T)
    objective = cp.norm(vertical_diffs, p=p) + cp.norm(horizontal_diffs, p=p)
    constraints = []
    for i in range(m):
        for j in range(n):
            if known_idxs[i, j]:
                constraints.append(U[i, j] == U_orig[i, j])
    return cp.Problem(cp.Minimize(objective), constraints)


def construct_f(nrows: int):
    f = np.zeros(shape=(nrows-1, nrows))
    for i in range(nrows-1):
            f[i, i] = -1
            f[i, i+1] = 1
    return f


def run_problem_5():
    np.random.seed(50)
    U_orig = plt.imread('/Users/maxwasserman/Desktop/conv_opt_probs/data/tv_img_interp.png')
    m, n = U_orig.shape
    known_idxs = np.random.uniform(0, 1, size=(m, n)) > 0.5

    l2_prob = image_interpolation(U_orig, known_idxs, p=2)
    l2_value = l2_prob.solve(solver='MOSEK')
    U_l2 = l2_prob.var_dict['U'].value

    l1_prob = image_interpolation(U_orig, known_idxs, p=1)
    l1_value = l1_prob.solve(solver='MOSEK')
    U_l1 = l1_prob.var_dict['U'].value

    fig, axs = plt.subplots(4, 1)
    imgs = [U_orig, known_idxs, U_l2, U_l1]
    txt = ['True Image', 'Known Indices', f'L2 Recon: Opt Val {l2_value:.1f}', f'L1 Recon Opt Val {l1_value:.1f}']
    for ax, im, t in zip(axs, imgs, txt):
        ax.imshow(im, cmap='binary') if 'Known' in t else ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.show()


### problem 6 ###
def relaxed_optimal_experiment(n: int, p: int, m: int, v: np.array):
    lambda_ = cp.Variable(p, name='fraction', nonneg=True)
    objective = (1/m) * cp.matrix_frac(X=np.eye(n), P=(v @ cp.diag(lambda_) @ cp.transpose(v)))
    constraints = [lambda_ @ np.ones(p) == 1, lambda_ >= 0]
    return cp.Problem(cp.Minimize(objective), constraints)


def project_onto_feasible_set(fraction: np.array, m):
    # fix found optimal fraction so it is a feasible point in the
    # original (strict) optimal experiment problem,
    # which has the constrain lambda_i \in integers, lambda^t 1 = m
    rounded_counts = np.rint(fraction * m)
    total_count = rounded_counts.sum()

    if rounded_counts.sum() > m:
        # too many selected: remove one from each until experiment until
        # correct total count
        i = 0
        while total_count > m:
            idx = i % p
            rounded_counts[idx] -= 1 if rounded_counts[idx] > 0 else 0
            total_count = rounded_counts.sum()
    elif rounded_counts.sum() < m:
        # too few selected: remove one from each until experiment until
        # correct total count
        i = 0
        while total_count < m:
            idx = i % p
            rounded_counts[idx] += 1
            total_count = rounded_counts.sum()

    return rounded_counts


def run_problem_6():
    # n: dim of params to be estimated, p: # available types of measurements,
    # m: total # of msmsts to be carried out
    n, p, m = 5, 20, 30
    np.random.seed(0)
    # columns are vi, the possible measurement vectors
    V = np.random.randn(n, p)

    relax_optimal_exp = relaxed_optimal_experiment(n, p, m, V)
    p_optimal_relax_optimal_exp = relax_optimal_exp.solve(solver="MOSEK")
    projected_counts = project_onto_feasible_set(relax_optimal_exp.var_dict['fraction'].value, m)
    suboptimal_objective = np.trace( np.linalg.inv(V @ np.diag(projected_counts) @ V.transpose()) )

    print(f'relaxed objective value {p_optimal_relax_optimal_exp:.5f} gives lower bound on p*')
    print(f'projected objective value {suboptimal_objective:.5f} gives upper bound on p*')
    print(f'thus p* in [{p_optimal_relax_optimal_exp:.5f}, {suboptimal_objective:.5f}]')
    suboptimality = suboptimal_objective - p_optimal_relax_optimal_exp
    print(f'we say the found the projected point is at most (projected obj val) - (relax obj val) = {suboptimality:.5f} suboptimal ')


### edx problem: Maximum likelihood estimation of an increasing nonnegative signal ###
def mle_edx(y: np.array, h: np.array, n: int, include_incr_nonneg: bool = False):
    x = cp.Variable(n, name='x')
    c = cp.Variable(len(h)+n-1, name='conv')
    constraints = [c == cp.reshape(cp.conv(h, x), (len(h)+n-1,))]
    if include_incr_nonneg:
        constraints.append(x[0] >= 0)
        for i in range(n-1):
            constraints.append(x[i] <= x[i+1])
    objective = cp.square(cp.norm(y-c[0:-3], p=2))
    return cp.Problem(cp.Minimize(objective), constraints)


def run_mle_edx():
    # problem data
    n = 100
    x_true = np.zeros(n)
    x_true[0:40], x_true[50], x_true[70:80], x_true[80] = 0.1, 2, 0.15, 1
    x_true = np.cumsum(x_true)
    # pass the increasing input through a moving - average
    #  filter and add Gaussian noise

    h = np.array([1, - 0.85, 0.7, - 0.3])
    k = len(h)
    y_hat = np.convolve(h, x_true)
    y = y_hat[0:-3] + np.random.randn(*y_hat[0:-3].shape)
    assert y.shape == x_true.shape
    """
        + [-0.43, -1.7,0.13,0.29,-1.1,1.2,1.2,-0.038,0.33,0.17,-0.19,0.73,-0.59,2.2,-0.14,0.11,1.1,
        0.059,-0.096,-0.83,0.29,-1.3,0.71,1.6,-0.69,0.86,1.3,-1.6,-1.4,0.57,-0.4,0.69,0.82,0.71,1.3,
        0.67,1.2,-1.2,-0.02,-0.16,-1.6,0.26,-1.1,1.4,-0.81,0.53,0.22,-0.92,-2.2,-0.059,-1,0.61,0.51,
        1.7,0.59,-0.64,0.38,-1,-0.02,-0.048,4.3e-05,
        -0.32,
        1.1,
        -1.9,
        0.43,
        0.9,
        0.73,
        0.58,
        0.04,
        0.68,
        0.57,
        -0.26,
        -0.38,
        -0.3,
        -1.5,
        -0.23,
        0.12,
        0.31,
        1.4,
        -0.35,
        0.62,
        0.8,
        0.94,
        -0.99,
        0.21,
        0.24,
        -1,
        -0.74,
        1.1,
        -0.13,
        0.39,
        0.088,
        -0.64,
        -0.56,
        0.44,
        -0.95,
        0.78,
        0.57,
        -0.82,
        -0.27]
        """
    mle_edx_prob_extra_constraints = mle_edx(y, h, n, include_incr_nonneg=True)
    mle_edx_prob = mle_edx(y, h, n, include_incr_nonneg=False)
    mle_edx_prob.solve(solver="MOSEK")
    mle_edx_prob_extra_constraints.solve(solver="MOSEK")
    x_raw, x_extra_constr = mle_edx_prob.var_dict['x'].value, mle_edx_prob_extra_constraints.var_dict['x'].value

    fig, axs = plt.subplots(nrows=1, ncols=1)
    t = np.arange(1, n+1)
    axs.plot(t, x_raw, label='x no contraints')
    axs.plot(t, x_extra_constr, label='x non-neg/decr contraints')
    axs.plot(t, x_true, label='true x')
    plt.legend()
    plt.show()

run_mle_edx()