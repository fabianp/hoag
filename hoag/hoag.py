import numpy as np
from numpy import array, asarray, float64, int32, zeros
from scipy import linalg
from scipy.optimize.lbfgsb import _lbfgsb
from scipy.sparse import linalg as splinalg


def hoag_lbfgs(
    h_func_grad, h_hessian, h_crossed, g_func_grad, x0, bounds=None,
    lambda0=0., disp=None, maxcor=10,
    maxiter=100, maxiter_inner=10000,
    only_fit=False,
    iprint=-1, maxls=20, tolerance_decrease='exponential',
    callback=None, verbose=0, epsilon_tol_init=1e-3, exponential_decrease_factor=0.9,
    projection=None  # LUCA added this to better deal with projections of hyperparameters
):
    """
    HOAG algorithm using L-BFGS-B in the inner optimization algorithm.

    Options
    -------
    eps : float
        Step size used for numerical approximation of the Jacobian.
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

    exact_epsilon = 1e-12
    if tolerance_decrease == 'exact':
        epsilon_tol = exact_epsilon
    else:
        epsilon_tol = epsilon_tol_init

    Bxk = None
    L_lambda = None
    g_func_old = np.inf

    if callback is not None:  # LUCA pass also hyper-gradient to callback
        callback(x, lambdak, np.zeros_like(lambdak), 0.)

    # n_eval, F = wrap_function(F, ())
    h_func, h_grad = h_func_grad(x, lambdak)
    norm_init = linalg.norm(h_grad)
    old_grads = []

    for it in range(1, maxiter):
        h_func, h_grad = h_func_grad(x, lambdak)
        n_iterations = 0
        task[:] = 'START'
        old_x = x.copy()
        while 1:
            pgtol_lbfgs = 1e-120
            factr = 1e-120  # / np.finfo(float).eps
            _lbfgsb.setulb(
                m, x, low_bnd, upper_bnd, nbd, h_func, h_grad,
                factr, pgtol_lbfgs, wa, iwa, task, iprint, csave, lsave,
                isave, dsave, maxls)
            task_str = task.tostring()
            if task_str.startswith(b'FG'):
                # minimization routine wants h_func and h_grad at the current x
                # Overwrite h_func and h_grad:
                h_func, h_grad = h_func_grad(x, lambdak)
                if linalg.norm(h_grad) / np.exp(np.min(lambdak)) < epsilon_tol * norm_init:
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
                if verbose > 1:
                    print('LBFGS decided finish!')
                    print(task_str)
                break
        else:
            pass

        if only_fit:
            break

        if verbose > 0:
            h_func, h_grad = h_func_grad(x, lambdak)
            print('inner level iterations: %s, inner objective %s, grad norm %s' % (n_iterations, h_func, linalg.norm(h_grad)))

        fhs = h_hessian(x, lambdak)
        B_op = splinalg.LinearOperator(
            shape=(x.size, x.size),
            matvec=lambda z: fhs(z))

        g_func, g_grad = g_func_grad(x, lambdak)
        if Bxk is None:
            Bxk = x.copy()
        residual_init = linalg.norm(g_grad)
        if verbose > 1:
            print('Inverting matrix with precision %s' % (epsilon_tol * residual_init))
        Bxk, success = splinalg.cg(B_op, g_grad, x0=Bxk, tol=epsilon_tol * residual_init, maxiter=maxiter_inner)
        if success != 0:
            print('CG did not converge to the desired precision')
        old_epsilon_tol = epsilon_tol
        if tolerance_decrease == 'quadratic':
            epsilon_tol = epsilon_tol_init / (it ** 2)
        elif tolerance_decrease == 'cubic':
            epsilon_tol = epsilon_tol_init / (it ** 3)
        elif tolerance_decrease == 'exponential':
            epsilon_tol *= exponential_decrease_factor
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
                L_lambda = old_grads[-1] / np.sqrt(len(lambdak))
            else:
                L_lambda = 1

        step_size = (1./L_lambda)

        old_lambdak = lambdak.copy()
        lambdak -= step_size * grad_lambda

        # projection
        if projection is None:  # LUCA as originally implemented
            lambdak[lambdak < -6] = -6
            lambdak[lambdak > 6] = 6
        else:
            lambdak = projection(lambdak)

        incr = linalg.norm(lambdak - old_lambdak)

        C = 0.25
        factor_L_lambda = 1.0
        if g_func <= g_func_old + C * epsilon_tol + \
                old_epsilon_tol * (C + factor_L_lambda) * incr - factor_L_lambda * (L_lambda) * incr * incr:
            L_lambda *= 0.95
            if verbose > 1:
                print('increased step size')
        elif g_func >= 1.2 * g_func_old:
            if verbose > 1:
                print('decrease step size')
            # decrease step size
            L_lambda *= 2
            lambdak = old_lambdak
            print('!!step size rejected!!', g_func, g_func_old)
            g_func_old, g_grad_old = g_func_grad(x, old_lambdak)
            # tighten tolerance
            epsilon_tol *= 0.5
        else:
            pass

        # if g_func - g_func_old > 0:
        #     raise ValueError
        norm_grad_lambda = linalg.norm(grad_lambda)
        if verbose > 0:
            print(('it %s, g: %s, incr: %s, sum lambda %s, epsilon: %s, ' +
                  'L: %s, norm grad_lambda: %s') %
                  (it, g_func, g_func - g_func_old, lambdak.sum(), epsilon_tol, L_lambda,
                   norm_grad_lambda))
        g_func_old = g_func

        if callback is not None:
            callback(x, lambdak, grad_lambda, step_size)

    task_str = task.tostring().strip(b'\x00').strip()
    if task_str.startswith(b'CONV'):
        warnflag = 0
    else:
        warnflag = 2

    return x, lambdak, warnflag
