
import numpy as np
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn import linear_model, cross_validation
from hoag import LogisticRegressionCV, MultiLogisticRegressionCV


def test_LogisticRegressionCV():
    bunch = fetch_20newsgroups_vectorized(subset="train")
    X = bunch.data
    y = bunch.target

    y[y < y.mean()] = -1
    y[y >= y.mean()] = 1
    Xt, Xh, yt, yh = cross_validation.train_test_split(
        X, y, test_size=.5, random_state=0)

    # compute the scores
    all_scores = []
    all_alphas = np.linspace(-12, 0, 5)
    for a in all_alphas:
        lr = linear_model.LogisticRegression(
            solver='lbfgs', C=np.exp(-a), fit_intercept=False, tol=1e-6,
            max_iter=100)
        lr.fit(Xt, yt)
        score_scv = linear_model.logistic._logistic_loss(
            lr.coef_.ravel(), Xh, yh, 0)
        all_scores.append(score_scv)
    all_scores = np.array(all_scores)

    best_alpha = all_alphas[np.argmin(all_scores)]

    clf = LogisticRegressionCV(verbose=True, max_iter=200)
    clf.fit(Xt, yt, Xh, yh)
    np.testing.assert_array_less(np.abs(clf.alpha_ - best_alpha), 0.5)


def test_MultiLogisticRegressionCV():
    bunch = fetch_20newsgroups_vectorized(subset="train")
    X = bunch.data
    y = bunch.target

    # y[y < y.mean()] = -1
    # y[y >= y.mean()] = 1
    Xt, Xh, yt, yh = cross_validation.train_test_split(
        X, y, test_size=.5, random_state=0)
    #
    # # compute the scores
    # all_scores = []
    # all_alphas = np.linspace(-12, 0, 5)
    # for a in all_alphas:
    #     lr = linear_model.LogisticRegression(
    #         solver='lbfgs', C=np.exp(-a), fit_intercept=False, tol=1e-6,
    #         max_iter=100)
    #     lr.fit(Xt, yt)
    #     score_scv = linear_model.logistic._logistic_loss(
    #         lr.coef_.ravel(), Xh, yh, 0)
    #     all_scores.append(score_scv)
    # all_scores = np.array(all_scores)
    #
    # best_alpha = all_alphas[np.argmin(all_scores)]
    #
    clf = MultiLogisticRegressionCV(verbose=True, max_iter=5)
    clf.fit(Xt, yt, Xh, yh)
    # np.testing.assert_array_less(np.abs(clf.alpha_ - best_alpha), 0.5)


if __name__ == '__main__':
    test_LogisticRegressionCV()