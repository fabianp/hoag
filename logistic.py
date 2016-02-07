import numpy as np
from numpy import array, asarray, float64, int32, zeros
from scipy import linalg
from scipy.optimize.lbfgsb import _check_unknown_options, _lbfgsb,\
    LbfgsInvHessProduct
from scipy.sparse import linalg as splinalg
from sklearn import linear_model
from sklearn.linear_model.logistic import _logistic_loss,\
    _logistic_loss_and_grad, _logistic_grad_hess



class LogisticRegressionCV(linear_model.base.BaseEstimator,
                           linear_model.base.LinearClassifierMixin):

    def __init__(
                 self, alpha0=0., tol=0.1, callback=None, verbose=False,
                 tolerance_decrease='exponential', max_iter=100):
        self.alpha0 = alpha0
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.tolerance_decrease = tolerance_decrease
        self.max_iter = max_iter

    def fit(self, Xt, yt, Xh, yh, callback=None):
        x0 = np.random.randn(Xt.shape[1])

        def h_func_grad(x, alpha):
            return _logistic_loss_and_grad(
                x, Xt, yt, np.exp(alpha[0]))

        def h_hessian(x, alpha):
            return _logistic_grad_hess(
                x, Xt, yt, np.exp(alpha[0]))[1]

        def g_func_grad(x, alpha):
            return _logistic_loss_and_grad(x, Xh, yh, 0)

        def h_crossed(x, alpha):
            return np.exp(alpha[0]) * x

        from hoag import _minimize_lbfgsb
        opt = _minimize_lbfgsb(
            h_func_grad, h_hessian, h_crossed, g_func_grad, x0,
            callback=callback,
            tolerance_decrease=self.tolerance_decrease,
            lambda0=[self.alpha0], maxiter=self.max_iter)

        # opt = _minimize_lbfgsb(
        #     h_func_grad, DE_DX, H, x0, callback=callback,
        #     tolerance_decrease=self.tolerance_decrease,
        #     lambda0=self.alpha0, maxiter=self.max_iter)

        self.coef_ = opt[0]
        self.alpha_ = opt[1]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_)

    def predict(self, X):
        return np.sign(self.decision_function(X))


