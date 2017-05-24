from functools import reduce

from hoag.logistic import _logistic_loss, LogisticRegressionCV
import sklearn.datasets as sk_dt
import numpy as np
from collections import OrderedDict

try:
    from tabulate import tabulate
except ImportError:
    print('Might want to install library "tabulate" for a better dictionary printing')
    tabulate = None


def as_list(obj):
    """
    Makes sure `obj` is a list or otherwise converts it to a list with a single element.

    :param obj:
    :return: A `list`
    """
    return obj if isinstance(obj, list) else [obj]


class Datasets:
    def __init__(self, train=None, validation=None, test=None):
        self.train = train
        self.validation = validation
        self.test = test
        self._lst = [train, validation, test]

    def setting(self):
        return {k: v.setting() if hasattr(v, 'setting') else None for k, v in vars(self).items()}

    def __getitem__(self, item):
        return self._lst[item]

    @staticmethod
    def from_list(list_of_datasets):
        train, valid, test = None, None, None
        train = list_of_datasets[0]
        if len(list_of_datasets) > 3:
            print('There are more then 3 Datasets here...')
            return list_of_datasets
        if len(list_of_datasets) > 1:
            test = list_of_datasets[-1]
            if len(list_of_datasets) == 3:
                valid = list_of_datasets[1]
        return Datasets(train, valid, test)


def _maybe_cast_to_scalar(what):
    return what[0] if len(what) == 1 else what


class Dataset:
    def __init__(self, data, target, sample_info_dicts=None, general_info_dict=None):
        """

        :param data: Numpy array containing data
        :param target: Numpy array containing targets
        :param sample_info_dicts: either an array of dicts or a single dict, in which case it is cast to array of
                                  dicts.
        :param general_info_dict: (optional) dictionary with further info about the dataset
        """
        self._tensor_mode = False

        self._data = data
        self._target = target
        if sample_info_dicts is None:
            sample_info_dicts = {}
        self.sample_info_dicts = np.array([sample_info_dicts] * self.num_examples) \
            if isinstance(sample_info_dicts, dict) else sample_info_dicts

        assert self.num_examples == len(self.sample_info_dicts)
        assert self.num_examples == self._shape(self._target)[0]

        self.general_info_dict = general_info_dict or {}

    def _shape(self, what):
        return what.get_shape().as_list() if self._tensor_mode else what.shape

    def setting(self):
        return {
            'num_examples': self.num_examples,
            'dim_data': self.dim_data,
            'dim_target': self.dim_target,
            'info': self.general_info_dict
        }

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    @property
    def num_examples(self):
        """

        :return: Number of examples in this dataset
        """
        return self._shape(self.data)[0]

    @property
    def dim_data(self):
        """

        :return: The data dimensionality as an integer, if input are vectors, or a tuple in the general case
        """
        return _maybe_cast_to_scalar(self._shape(self.data)[1:])

    @property
    def dim_target(self):
        """

        :return: The target dimensionality as an integer, if targets are vectors, or a tuple in the general case
        """
        shape = self._shape(self.target)
        return 1 if len(shape) == 1 else _maybe_cast_to_scalar(shape[1:])


def redivide_data(datasets, partition_proportions=None, shuffle=False, filters=None, maps=None):
    """
    Function that redivides datasets. Can be use also to shuffle or filter or map examples.

    :param datasets: original datasets, instances of class Dataset (works with get_data and get_targets for
    compatibility with mnist datasets
    :param partition_proportions: (optional, default None)  list of fractions that can either sum up to 1 or less
    then one, in which case one additional partition is created with proportion 1 - sum(partition proportions).
    If None it will retain the same proportion of samples found in datasets
    :param shuffle: (optional, default False) if True shuffles the examples
    :param filters: (optional, default None) filter or list of filters: functions with signature
    (data, target, index) -> boolean (accept or reject the sample)
    :param maps: (optional, default None) map or list of maps: functions with signature
    (data, target, index) ->  (new_data, new_target) (maps the old sample to a new one, possibly also to more
    than one sample, for data augmentation)
    :return: a list of datasets of length equal to the (possibly augmented) partition_proportion
    """
    import scipy.sparse as sp

    def stack_or_concat(list_of_arays):
        func = np.concatenate if list_of_arays[0].ndim == 1 else np.vstack
        return func(list_of_arays)

    def vstack(lst):
        return sp.vstack(lst) if isinstance(lst[0], sp.csr.csr_matrix) else np.vstack(lst)

    all_data = vstack([d.data for d in datasets])
    all_labels = stack_or_concat([d.target for d in datasets])

    all_infos = np.concatenate([d.sample_info_dicts for d in datasets])

    N = all_data.shape[0]

    if partition_proportions:  # argument check
        partition_proportions = list([partition_proportions] if isinstance(partition_proportions, float)
                                     else partition_proportions)
        sum_proportions = sum(partition_proportions)
        assert sum_proportions <= 1, "partition proportions must sum up to at most one: %d" % sum_proportions
        if sum_proportions < 1.: partition_proportions += [1. - sum_proportions]
    else:
        partition_proportions = [1. * d.data.shape[0] / N for d in datasets]

    if shuffle:
        if isinstance(all_data, sp.csr.csr_matrix): raise NotImplementedError()
        # if sk_shuffle:  # TODO this does not work!!! find a way to shuffle these matrices while
        #  ....
        permutation = np.arange(all_data.shape[0])
        np.random.shuffle(permutation)

        all_data = all_data[permutation]
        all_labels = np.array(all_labels[permutation])
        all_infos = np.array(all_infos[permutation])

    if filters:
        if isinstance(all_data, sp.csr.csr_matrix): raise NotImplementedError()
        filters = as_list(filters)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for fiat in filters:
            data_triple = [xy for i, xy in enumerate(data_triple) if fiat(xy[0], xy[1], xy[2], i)]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    if maps:
        if isinstance(all_data, sp.csr.csr_matrix): raise NotImplementedError()
        maps = as_list(maps)
        data_triple = [(x, y, d) for x, y, d in zip(all_data, all_labels, all_infos)]
        for _map in maps:
            data_triple = [_map(xy[0], xy[1], xy[2], i) for i, xy in enumerate(data_triple)]
        all_data = np.vstack([e[0] for e in data_triple])
        all_labels = np.vstack([e[1] for e in data_triple])
        all_infos = np.vstack([e[2] for e in data_triple])

    N = all_data.shape[0]
    assert N == all_labels.shape[0]

    calculated_partitions = reduce(
        lambda v1, v2: v1 + [sum(v1) + v2],
        [int(N * prp) for prp in partition_proportions],
        [0]
    )
    calculated_partitions[-1] = N

    print('datasets.redivide_data:, computed partitions numbers -',
          calculated_partitions, 'len all', N, end=' ')

    new_general_info_dict = {}
    for data in datasets:
        new_general_info_dict = {**new_general_info_dict, **data.general_info_dict}

        new_datasets = [
            Dataset(data=all_data[d1:d2], target=all_labels[d1:d2], sample_info_dicts=all_infos[d1:d2],
                    general_info_dict=new_general_info_dict)
            for d1, d2 in zip(calculated_partitions, calculated_partitions[1:])
        ]

        print('DONE')

        return new_datasets


def generate_multiclass_dataset(n_samples=100, n_features=10,
                                n_informative=5, n_redundant=3, n_repeated=2,
                                n_classes=2, n_clusters_per_class=2,
                                weights=None, flip_y=0.01, class_sep=1.0,
                                hypercube=True, shift=0.0, scale=1.0,
                                shuffle=True, random_state=None, hot_encoded=True, partitions_proportions=None,
                                negative_labels=-1.):
    X, y = sk_dt.make_classification(n_samples=n_samples, n_features=n_features,
                                     n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated,
                                     n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
                                     weights=weights, flip_y=flip_y, class_sep=class_sep,
                                     hypercube=hypercube, shift=shift, scale=scale,
                                     shuffle=True, random_state=random_state)
    if hot_encoded:
        def to_one_hot_enc(seq):
            da_max = np.max(seq) + 1

            def create_and_set(_p):
                _tmp = np.zeros(da_max)
                _tmp[_p] = 1
                return _tmp

            return np.array([create_and_set(_v) for _v in seq])

        y = to_one_hot_enc(y)
    else:
        y[y == 0] = negative_labels
    res = Dataset(data=np.array(X, dtype=np.float32), target=np.array(y, dtype=np.float32),
                  general_info_dict={'n_informative': n_informative, 'n_redundant': n_redundant,
                                     'n_repeated': n_repeated,
                                     'n_classes': n_classes, 'n_clusters_per_class': n_clusters_per_class,
                                     'weights': weights, 'flip_y': flip_y, 'class_sep': class_sep,
                                     'hypercube': hypercube, 'shift': shift, 'scale': scale,
                                     'shuffle': True, 'random_state': random_state})
    np.random.seed(random_state)
    if partitions_proportions:
        res = redivide_data([res], shuffle=shuffle, partition_proportions=partitions_proportions)
        res = Datasets.from_list(res)
    return res


def hoag_fit(datasets, max_iter=100, alpha0=0., verbose=2,
             projection=None, do_print=False, hyper_step_mul=1.):
    tr_sup = (datasets.train.data, datasets.train.target)
    # n_tr = datasets.train.num_examples
    val_sup = (datasets.validation.data, datasets.validation.target)
    # n_val = n_tr = datasets.validation.num_examples
    tst_sup = (datasets.test.data, datasets.test.target)

    clf = LogisticRegressionCV(verbose=verbose, max_iter=max_iter, alpha0=alpha0)

    log_dict = OrderedDict([['training error', lambda: _logistic_loss(clf.coef_, tr_sup[0], tr_sup[1],
                                                           clf.alpha_)],
                ['validation error', lambda: _logistic_loss(clf.coef_, val_sup[0],
                                                             val_sup[1], alpha=0.)],
                ['test error', lambda: _logistic_loss(clf.coef_, tst_sup[0],
                                                       tst_sup[1], alpha=0.)],
                ['validation accuracy', lambda: clf.accuracy(val_sup[0], val_sup[1])],
                ['test accuracy', lambda: clf.accuracy(tst_sup[0], tst_sup[1])],
                ['alpha', lambda: clf.alpha_],
                ['der alpha', lambda: clf.der_alpha_],
                ['step size', lambda: clf.step_size]])
    all_steps_log = OrderedDict([[k, []] for k in log_dict])

    stp = 0

    def callback(x, alpha, der_alpha, step_size):
        nonlocal stp
        clf.coef_ = np.array(x)
        clf.alpha_ = np.array(alpha)
        clf.der_alpha_ = der_alpha
        clf.step_size = step_size

        print()
        print('Log step', stp)
        [all_steps_log[k].append(v()) for k, v in log_dict.items()]
        if tabulate:
            print(tabulate([(k, v()) for k, v in log_dict.items()]))
        else:
            for k, v in log_dict.items(): print(k, v())

        stp += 1

    clf.fit(tr_sup[0], tr_sup[1], val_sup[0], val_sup[1], callback=callback, projection=projection)

    return clf, all_steps_log
