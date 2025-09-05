from typing import List, Tuple, Any
from scipy.stats._distn_infrastructure import rv_continuous
from scipy.interpolate import interp1d
import numpy as np


class CensoredCDFEstimator(rv_continuous):
    def __init__(self, support: Tuple[float, float] = (-np.inf, np.inf),
                 momtype=1,
                 a=None,
                 b=None,
                 xtol=1e-14,
                 badvalue=None,
                 name=None,
                 longname=None,
                 shapes=None,
                 extradoc=None,
                 seed=None):

        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue,
                         name=name, longname=longname, shapes=shapes, seed=seed)
        self.support_ = support

    def fit(self, intervals: List[Tuple[float, float]],
            max_iter=50, tol=1e-4, verbose=False, verbose_interval=1):

        self.qp_ints_ = self._extract_disjoint_intervals(intervals)

        alphas = np.fromfunction(
            np.vectorize(lambda i, j: self._if_sub_interval(intervals[i], self.qp_ints_[j])),
            shape=(len(intervals), len(self.qp_ints_)), dtype=int)

        self.qp_probs_ = np.ones(len(self.qp_ints_)) / len(self.qp_ints_)

        for iteration in range(max_iter):
            cur_probs = self.qp_probs_.copy()
            ms = (alphas * cur_probs) / (alphas @ cur_probs).reshape(-1, 1)
            self.qp_probs_ = ms.sum(0) / ms.sum()
            diff = np.linalg.norm(cur_probs - self.qp_probs_)
            if verbose and iteration % verbose_interval == 0:
                print(f"Iteration: {iteration}, Probs diff l2-norm: {round(diff, int(2 - np.log10(tol)))}")
            if diff < tol:
                break

        if len(self.qp_ints_[:, 0]) >= 2:
            self._cdf_interpolator = interp1d(self.qp_ints_[:, 0], np.cumsum(self.qp_probs_),
                                              bounds_error=False, fill_value=(0.0, 1.0))
        else:
            self._cdf_interpolator = lambda x: np.clip(x, 0, 1)

        return self

    def _cdf(self, x: Any, *args, **kwargs):
        return self._cdf_interpolator(x)

    def support(self):
        return self.support_

    @staticmethod
    def _if_sub_interval(interval1: Tuple[float, float], interval2: Tuple[float, float]):
        return (interval1[0] <= interval2[0]) and (interval1[1] >= interval2[1])

    @staticmethod
    def _extract_disjoint_intervals(intervals: List[Tuple[float, float]]):
        assert len(intervals) > 0, "At least one interval should be provided"
        assert all(interval[0] <= interval[1] for interval in intervals)
        lefts, rights = zip(*intervals)
        sort_endpoints = sorted([(left, "L") for left in lefts] + [(right, "R") for right in rights])
        disjoint_intervals = []
        for (ep, next_ep) in zip(sort_endpoints[:-1], sort_endpoints[1:]):
            if ep[1] == "L" and next_ep[1] == "R":
                disjoint_intervals.append((ep[0], next_ep[0]))
        return np.array(disjoint_intervals)
