import numpy as np
from numpy import array, asarray, float64, int32, zeros
from scipy import linalg
from scipy.optimize.lbfgsb import _lbfgsb
from scipy.sparse import linalg as splinalg


def hoag_lbfgs(
    h_func_grad, h_hessian, h_crossed, g_func_grad, x0, bounds=None,
    lambda0=0., disp=None, maxcor=10, ftol=1e-24,
    maxiter=100, only_fit=False,
    iprint=-1, maxls=20, tolerance_decrease='exponential',
    callback=None):
    """
    HOAG algorithm using L-BFGS-B in the inner optimization algorithm.

    Options
    -------
    eps : float
        Step size used for numerical approximation of the jacobian.
    disp : int
        Set to True to print convergence messages.
    maxfun : int
        Maximum number of function evaluations.
    maxiter : int
        Maximum number of iterations.
    maxls : int, optional
        Maximum number of line search steps (per iteration). Default is 20.
    """
    m = maxcor
    factr = ftol / np.finfo(float).eps
    lambdak = lambda0

    x0 = asarray(x0).ravel()
    n, = x0.shape

    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')
    # unbounded variables must use None, not +-inf, for optimizer to work properly
    bounds = [(None if l == -np.inf else l, None if u == np.inf else u) for l, u in bounds]

    if disp is not None:
        if disp == 0:
            iprint = -1
        else:
            iprint = disp

    nbd = zeros(n, int32)
    low_bnd = zeros(n, float64)
    upper_bnd = zeros(n, float64)
    bounds_map = {(None, None): 0,
                  (1, None): 1,
                  (1, 1): 2,
                  (None, 1): 3}
    for i in range(0, n):
        l, u = bounds[i]
        if l is not None:
            low_bnd[i] = l
            l = 1
        if u is not None:
            upper_bnd[i] = u
            u = 1
        nbd[i] = bounds_map[l, u]

    if not maxls > 0:
        raise ValueError('maxls must be positive.')

    x = array(x0, float64)
    wa = zeros(2*m*n + 5*n + 11*m*m + 8*m, float64)
    iwa = zeros(3*n, int32)
    task = zeros(1, 'S60')
    csave = zeros(1, 'S60')
    lsave = zeros(4, int32)
    isave = zeros(44, int32)
    dsave = zeros(29, float64)

    task[:] = 'START'

    epsilon_tol_init = .1
    exact_epsilon = 1e-12
    if tolerance_decrease == 'exact':
        epsilon_tol = exact_epsilon
    else:
        epsilon_tol = epsilon_tol_init

    Bxk = None
    L_lambda = None
    g_func_old = np.inf

    if callback is not None:
        callback(x, lambdak)

    # n_eval, F = wrap_function(F, ())
    h_func, g = h_func_grad(x, lambdak)
    norm_init = linalg.norm(g)
    old_grads = []

    for it in range(1, maxiter):
        h_func, h_grad = h_func_grad(x, lambdak)
        # print(linalg.norm(h_grad), epsilon_tol)
        n_iterations = 0
        while 1:
            pgtol_lbfgs = 1e-124 # exact_epsilon
            factr = 1e-124 # exact_epsilon
            # x, h_func, h_grad, wa, iwa, task, csave, lsave, isave, dsave = \
            _lbfgsb.setulb(
                m, x, low_bnd, upper_bnd, nbd, h_func, h_grad,
                factr, pgtol_lbfgs, wa, iwa, task, iprint, csave, lsave,
                isave, dsave, maxls)
            task_str = task.tostring()
            if task_str.startswith(b'FG'):
                # minimization routine wants h_func and h_grad at the current x
                # Overwrite h_func and h_grad:
                h_func_old = h_func
                h_func, h_grad = h_func_grad(x, lambdak)
                if linalg.norm(h_grad) < epsilon_tol:
                    print(linalg.norm(h_grad))
                    # this one is finished
                    break

            elif task_str.startswith(b'NEW_X'):
                # new iteration
                if n_iterations > 1e6:
                    task[:] = 'STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT'
                    print('ITERATIONS EXCEEDS LIMIT')
                else:
                    n_iterations += 1
            else:
                print('LBFGS decided finish!')
                print(task_str)
                break
                task[:] = 'START'
                # 1/0
                # break
        else:
            pass
            # print('Skipped LBFGS')

        if only_fit:
            break

        if tolerance_decrease == 'exact':
            print('Norm grad after lbfgs: %s' % linalg.norm(h_grad))

        fhs = h_hessian(x, lambdak)
        B_op = splinalg.LinearOperator(
            shape=(x.size, x.size),
            matvec=lambda z: fhs(z))

        g_func, g_grad = g_func_grad(x, lambdak)
        Bxk, success = splinalg.cg(B_op, g_grad, x0=Bxk, tol=epsilon_tol)
        if success is False:
            raise ValueError

        # .. update hyperparameters ..
        grad_lambda = - h_crossed(x, lambdak).dot(Bxk)
        old_grads.append(linalg.norm(grad_lambda))

        old_lambdak = lambdak
        # pk = 0.8 * grad_lambda + 0.2 * old_grad_lambda
        pk = grad_lambda

        if L_lambda is None:
            L_lambda = old_grads[-1]

        step_size = (1./L_lambda)

        tmp = lambdak - step_size * pk
        lambdak = tmp

        # .. decrease accuracy ..

        # .. decrease accuracy ..
        old_epsilon_tol = epsilon_tol
        if tolerance_decrease == 'quadratic':
            epsilon_tol = epsilon_tol_init / ((it) ** 2.)
        elif tolerance_decrease == 'cubic':
            epsilon_tol = epsilon_tol_init / ((it) ** 3.)
        elif tolerance_decrease == 'exponential':
            epsilon_tol *= 0.5
        elif tolerance_decrease == 'exact':
            epsilon_tol = 1e-24
        else:
            raise NotImplementedError

        epsilon_tol = max(epsilon_tol, exact_epsilon)
        incr = linalg.norm(lambdak - old_lambdak)

        if g_func - epsilon_tol <= g_func_old + old_epsilon_tol + \
                old_epsilon_tol * incr - (L_lambda / 2.) * incr * incr:
            # increase step size
            L_lambda *= .8
            print('increased step size')
        # elif g_func <= g_func_old:
        #     # do nothing
        #     pass
        else:
            print('decrease step size')
            # decrease step size
            if L_lambda < 1e3:
                L_lambda *= 1.2

        norm_lambda = linalg.norm(pk)
        print(('it %s, pk: %s, grad h_func: %s, lambda %s, epsilon: %s, ' +
              'L: %s, grad_lambda: %s') %
              (it, norm_lambda, linalg.norm(h_grad), lambdak, epsilon_tol, L_lambda,
               grad_lambda))
        g_func_old = g_func

        if callback is not None:
            callback(x, lambdak)

    task_str = task.tostring().strip(b'\x00').strip()
    if task_str.startswith(b'CONV'):
        warnflag = 0
    elif n_iterations > maxiter:
        warnflag = 1
    else:
        warnflag = 2

    return x, lambdak, warnflag
