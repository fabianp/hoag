import numpy as np
from scipy import sparse
from scipy.special import logsumexp
from sklearn import linear_model
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from sklearn.preprocessing import LabelBinarizer
from .hoag import hoag_lbfgs


class MultiLogisticRegressionCV(linear_model._base.BaseEstimator,
                           linear_model._base.LinearClassifierMixin):

    def __init__(self, alpha0=None, tol=0.1, callback=None, verbose=0,
                 tolerance_decrease='exponential', max_iter=10):
        self.alpha0 = alpha0
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.tolerance_decrease = tolerance_decrease
        self.max_iter = max_iter

    def fit(self, Xt, yt, Xh, yh, callback=None):
        lbin = LabelBinarizer()
        lbin.fit(yt)
        Yt_multi = lbin.transform(yt)
        Yh_multi = lbin.transform(yh)
        sample_weight_train = np.ones(Xt.shape[0])
        sample_weight_test = np.ones(Xh.shape[0])


        if Yt_multi.shape[1] == 1:
            Yt_multi = np.hstack([1 - Yt_multi, Yt_multi])
            Yh_multi = np.hstack([1 - Yh_multi, Yh_multi])
            print('warning: only two classes detected')

        n_classes = Yt_multi.shape[1]
        n_features = Xt.shape[1]

        if self.alpha0 is None:
            self.alpha0 = np.zeros(n_classes * n_features)  # if not np.all(np.unique(yt) == np.array([-1, 1])):
        #     raise ValueError
        x0 = np.zeros(n_features * n_classes)

        # assert x0.size == self.alpha0.size

        def h_func_grad(x, alpha):
            # x = x.reshape((-1,Yt_multi.shape[1]))
            return _multinomial_loss_grad(
                x, Xt, Yt_multi, np.exp(alpha), sample_weight_train)[:2]

        def h_hessian(x, alpha):
            # x = x.reshape((-1,Yt_multi.shape[1]))
            return _multinomial_grad_hess(
                x, Xt, Yt_multi, np.exp(alpha), sample_weight_train)[1]

        def g_func_grad(x, alpha):
            # x = x.reshape((-1,Yt_multi.shape[1]))
            return _multinomial_loss_grad(
                x, Xh, Yh_multi, np.zeros(alpha.size),
                sample_weight_test)[:2]

        def h_crossed(x, alpha):
            # return x.reshape((n_classes, -1)) * alpha
            # x = x.reshape((-1,Yt_multi.shape[1]))
            tmp = np.exp(alpha) * x
            return sparse.dia_matrix(
                (tmp, 0),
                shape=(n_features * n_classes, n_features * n_classes))

        opt = hoag_lbfgs(
            h_func_grad, h_hessian, h_crossed, g_func_grad, x0,
            callback=callback,
            tolerance_decrease=self.tolerance_decrease,
            lambda0=self.alpha0, maxiter=self.max_iter,
            verbose=self.verbose)

        self.coef_ = opt[0]
        self.alpha_ = opt[1]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_)

    def predict(self, X):
        return np.sign(self.decision_function(X))



class MultiLogisticRegression(linear_model._base.BaseEstimator,
                           linear_model._base.LinearClassifierMixin):

    def __init__(self, alpha0=None, tol=0.1, callback=None, verbose=0,
                 tolerance_decrease='exponential', max_iter=10):
        self.alpha0 = alpha0
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.tolerance_decrease = tolerance_decrease
        self.max_iter = max_iter

    def fit(self, Xt, yt, Xh, yh, callback=None):
        lbin = LabelBinarizer()
        lbin.fit(yt)
        Yt_multi = lbin.transform(yt)
        Yh_multi = lbin.transform(yh)
        sample_weight_train = np.ones(Xt.shape[0])
        sample_weight_test = np.ones(Xh.shape[0])

        if Yt_multi.shape[1] == 1:
            Yt_multi = np.hstack([1 - Yt_multi, Yt_multi])
            Yh_multi = np.hstack([1 - Yh_multi, Yh_multi])
            print('warning: only two classes detected')

        n_classes = Yt_multi.shape[1]
        n_features = Xt.shape[1]

        # if not np.all(np.unique(yt) == np.array([-1, 1])):
        #     raise ValueError
        x0 = np.zeros(n_features * n_classes)

        # assert x0.size == self.alpha0.size

        def h_func_grad(x, alpha):
            # x = x.reshape((-1,Yt_multi.shape[1]))
            return _multinomial_loss_grad(
                x, Xt, Yt_multi, np.exp(alpha), sample_weight_train)[:2]

        def h_hessian(x, alpha):
            # x = x.reshape((-1,Yt_multi.shape[1]))
            return _multinomial_grad_hess(
                x, Xt, Yt_multi, np.exp(alpha), sample_weight_train)[1]

        def g_func_grad(x, alpha):
            # x = x.reshape((-1,Yt_multi.shape[1]))
            return _multinomial_loss_grad(
                x, Xh, Yh_multi, np.zeros(alpha.size),
                sample_weight_test)[:2]

        def h_crossed(x, alpha):
            # return x.reshape((n_classes, -1)) * alpha
            # x = x.reshape((-1,Yt_multi.shape[1]))
            tmp = np.exp(alpha) * x
            return sparse.dia_matrix(
                (tmp, 0),
                shape=(n_features * n_classes, n_features * n_classes))

        opt = hoag_lbfgs(
            h_func_grad, h_hessian, h_crossed, g_func_grad, x0,
            callback=callback,
            tolerance_decrease=self.tolerance_decrease,
            lambda0=self.alpha0, maxiter=self.max_iter,
            verbose=self.verbose, only_fit=True)

        self.coef_ = opt[0]
        self.alpha_ = opt[1]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_)

    def predict(self, X):
        return np.sign(self.decision_function(X))

### The following is adapted from scikit-learn

L2_REG = 0

def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.
        If not provided, then each sample is given unit weight.

    Returns
    -------
    loss : float
        Multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities.

    w : ndarray, shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    alpha = alpha.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * (alpha * w * w).sum()
    p = np.exp(p, p)
    return loss, p, w


def _multinomial_loss_grad(w, X, Y, alpha, sample_weight):
    """Computes the multinomial loss, gradient and class probabilities.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.

    Returns
    -------
    loss : float
        Multinomial loss.

    grad : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    p : ndarray, shape (n_samples, n_classes)
        Estimated class probabilities

    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = (w.size == n_classes * (n_features + 1))
    grad = np.zeros((n_classes, n_features + bool(fit_intercept)))
    alpha = alpha.reshape(n_classes, -1)
    loss, p, w = _multinomial_loss(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    diff = sample_weight * (p - Y)
    grad[:, :n_features] = safe_sparse_dot(diff.T, X)
    grad[:, :n_features] += alpha * w
    if fit_intercept:
        grad[:, -1] = diff.sum(axis=0)
    return loss, grad.ravel(), p


def _multinomial_grad_hess(w, X, Y, alpha, sample_weight):
    """
    Computes the gradient and the Hessian, in the case of a multinomial loss.

    Parameters
    ----------
    w : ndarray, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    Y : ndarray, shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.

    alpha : float
        Regularization parameter. alpha is equal to 1 / C.

    sample_weight : array-like, shape (n_samples,) optional
        Array of weights that are assigned to individual samples.

    Returns
    -------
    grad : array, shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Ravelled gradient of the multinomial loss.

    hessp : callable
        Function that takes in a vector input of shape (n_classes * n_features)
        or (n_classes * (n_features + 1)) and returns matrix-vector product
        with hessian.

    References
    ----------
    Barak A. Pearlmutter (1993). Fast Exact Multiplication by the Hessian.
        http://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf
    """
    n_features = X.shape[1]
    n_classes = Y.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))

    # `loss` is unused. Refactoring to avoid computing it does not
    # significantly speed up the computation and decreases readability
    loss, grad, p = _multinomial_loss_grad(w, X, Y, alpha, sample_weight)
    sample_weight = sample_weight[:, np.newaxis]
    alpha = alpha.reshape(n_classes, -1)


    # Hessian-vector product derived by applying the R-operator on the gradient
    # of the multinomial loss function.
    def hessp(v):
        v = v.reshape(n_classes, -1)
        if fit_intercept:
            inter_terms = v[:, -1]
            v = v[:, :-1]
        else:
            inter_terms = 0
        # r_yhat holds the result of applying the R-operator on the multinomial
        # estimator.
        r_yhat = safe_sparse_dot(X, v.T)
        r_yhat += inter_terms
        r_yhat += (-p * r_yhat).sum(axis=1)[:, np.newaxis]
        r_yhat *= p
        r_yhat *= sample_weight
        hessProd = np.zeros((n_classes, n_features + bool(fit_intercept)))
        hessProd[:, :n_features] = safe_sparse_dot(r_yhat.T, X)
        hessProd[:, :n_features] += v * alpha
        if fit_intercept:
            raise ValueError
            hessProd[:, -1] = r_yhat.sum(axis=0)
        return hessProd.ravel()

    return grad, hessp
