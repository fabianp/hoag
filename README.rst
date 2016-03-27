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

Usage tips
----------

Standardize features of the input data such that each feature has unit variance. This makes the Hessian better conditioned. This can be done using e.g. scikit-learn's StandardScaler.
