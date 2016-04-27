import numpy as np
from numpy import array, asarray, float64, int32, zeros
from scipy import linalg
from scipy.optimize.lbfgsb import _lbfgsb
from scipy.sparse import linalg as splinalg


def hoag_lbfgs(
    h_func_grad, h_hessian, h_crossed, g_func_grad, x0, bounds=None,
    lambda0=0., disp=None, maxcor=10, ftol=1e-24,
    maxiter=100, maxiter_inner=1e6,
    only_fit=False,
    iprint=-1, maxls=20, tolerance_decrease='exponential',
    callback=None, verbose=0):
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
    lambdak = lambda0
    print(verbose)
    if verbose > 0:
        print('started hoag')

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

    epsilon_tol_init = 0.01
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
    h_func, h_grad = h_func_grad(x, lambdak)
    residual_init = None
    norm_init = linalg.norm(h_grad)
    old_grads = []

    for it in range(1, maxiter):
        h_func, h_grad = h_func_grad(x, lambdak)
        n_iterations = 0
        task[:] = 'START'
        while 1:
            pgtol_lbfgs = 1e-12
            factr = 1e-12  # / np.finfo(float).eps
            _lbfgsb.setulb(
                m, x, low_bnd, upper_bnd, nbd, h_func, h_grad,
                factr, pgtol_lbfgs, wa, iwa, task, iprint, csave, lsave,
                isave, dsave, maxls)
            task_str = task.tostring()
            if task_str.startswith(b'FG'):
                # minimization routine wants h_func and h_grad at the current x
                # Overwrite h_func and h_grad:
                h_func, h_grad = h_func_grad(x, lambdak)
                if linalg.norm(h_grad) < epsilon_tol * norm_init:
                    if verbose > 0:
                        print('Ended inner with %s iterations and %s h_grad norm' % (n_iterations, linalg.norm(h_grad)))
                    # this one is finished
                    break

            elif task_str.startswith(b'NEW_X'):
                # new iteration
                if n_iterations > maxiter_inner:
                    task[:] = 'STOP: TOTAL NO. of ITERATIONS EXCEEDS LIMIT'
                    print('ITERATIONS EXCEEDS LIMIT')
                    continue
                    # break
                else:
                    n_iterations += 1
            else:
                if verbose > 0:
                    print('LBFGS decided finish!')
                    print(task_str)
                break
                # 1/0
                # break
        else:
            pass
            # print('Skipped LBFGS')

        if only_fit:
            break

        if verbose > 0:
            h_func, h_grad = h_func_grad(x, lambdak)
            print('inner level iterations: %s, objective %s, grad norm %s' % (n_iterations, h_func, linalg.norm(h_grad)))

        if tolerance_decrease == 'exact':
            if verbose > 0:
                print('Norm grad after lbfgs: %s' % linalg.norm(h_grad))

        fhs = h_hessian(x, lambdak)
        B_op = splinalg.LinearOperator(
            shape=(x.size, x.size),
            matvec=lambda z: fhs(z))

        g_func, g_grad = g_func_grad(x, lambdak)
        if Bxk is None:
            Bxk = x.copy()
        if residual_init is None:
            # residual_init = 1.
            residual_init = linalg.norm(g_grad)
        print('Inverting matrix with precision %s' % (epsilon_tol * residual_init))
        Bxk, success = splinalg.cg(B_op, g_grad, x0=Bxk, tol=epsilon_tol * residual_init, maxiter=None)
        if success != 0:
            raise ValueError

        # .. decrease accuracy ..
        old_epsilon_tol = epsilon_tol
        if tolerance_decrease == 'quadratic':
            epsilon_tol = epsilon_tol_init / (it ** 2)
        elif tolerance_decrease == 'cubic':
            epsilon_tol = epsilon_tol_init / (it ** 3)
        elif tolerance_decrease == 'exponential':
            epsilon_tol *= 0.5
        elif tolerance_decrease == 'exact':
            epsilon_tol = 1e-24
        else:
            raise NotImplementedError

        epsilon_tol = max(epsilon_tol, exact_epsilon)
        # .. update hyperparameters ..
        grad_lambda = - h_crossed(x, lambdak).dot(Bxk)
        if linalg.norm(grad_lambda) == 0:
            # increase tolerance
            if verbose > 0:
                print('too low tolerance %s, moving to next iteration' % epsilon_tol)
            continue
        old_grads.append(linalg.norm(grad_lambda))

        if L_lambda is None:
            if old_grads[-1] > 1e-3:
                # make sure we are not selecting a step size that is too smal
                L_lambda = old_grads[-1]
            else:
                L_lambda = 1

        step_size = (1./L_lambda)

        old_lambdak = lambdak
        lambdak -= step_size * grad_lambda
        incr = linalg.norm(lambdak - old_lambdak)

        if g_func - epsilon_tol <= g_func_old + old_epsilon_tol + \
                old_epsilon_tol * incr - (L_lambda / 2.) * incr * incr:
            # increase step size
            L_lambda *= .8
            if verbose > 0:
                print('increased step size')
        else:
            if verbose > 0:
                print('decrease step size')
            # decrease step size
            if L_lambda < 1e3:
                L_lambda *= 2.0

        norm_grad_lambda = linalg.norm(grad_lambda)
        if verbose > 0:
            print(('it %s, grad, sum lambda %s, epsilon: %s, ' +
                  'L: %s, norm grad_lambda: %s') %
                  (it, lambdak.sum(), epsilon_tol, L_lambda,
                   norm_grad_lambda))
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
