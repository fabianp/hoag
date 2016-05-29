.. image:: https://travis-ci.org/fabianp/hoag.svg?branch=master
    :target: https://travis-ci.org/fabianp/hoag

HOAG
====
Hyperparameter optimization with approximate gradient

.. image:: https://raw.githubusercontent.com/fabianp/hoag/master/doc/comparison_ho_real_sim.png
   :scale: 50 %


Depends
-------

  * scikit-learn 0.16

Usage
-----

This package exports a LogisticRegressionCV class which automatically estimates the L2 regularization of logistic regression. As other scikit-learn objects, it has a .fit and .predict method. However, unlike scikit-learn objects, the .fit method takes 4 arguments consisting of the train set and the test set. For example:

    >>> from hoag import LogisticRegressionCV
    >>> clf = LogisticRegressionCV()
    >>> clf.fit(X_train, y_train, X_test, y_test)

where X_train, y_train, X_test, y_test are numpy arrays representing the train and test set, respectively.

Usage tips
----------

Standardize features of the input data such that each feature has unit variance. This makes the Hessian better conditioned. This can be done using e.g. scikit-learn's StandardScaler.
