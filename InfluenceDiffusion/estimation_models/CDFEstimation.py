from typing import List, Tuple, Any
from scipy.stats._distn_infrastructure import rv_continuous
from scipy.interpolate import interp1d
import numpy as np

__all__ = ["CensoredCDFEstimator"]


class CensoredCDFEstimator(rv_continuous):
    """
    This class implements the Turnbull estimator of the cumulative distribution function (CDF)
    from a collection of censored intervals within which sample points are observed.
    See: Turnbull, Bruce W. (1976).
    “The Empirical Distribution Function with Arbitrarily Grouped, Censored and Truncated Data.”
    Journal of the Royal Statistical Society. Series B (Methodological) 38(3), 290–295.

    Parameters
    ----------
    support : tuple of float, optional
        Explicit support (min, max) for the estimator. Default is (-np.inf, np.inf).
    a : float, optional
        Lower bound of the distribution. Default is None.
    b : float, optional
        Upper bound of the distribution. Default is None.
    momtype : int, optional
        Moment type used by `rv_continuous`. Default is 1.
    xtol : float, optional
        Tolerance for calculations in `rv_continuous`. Default is 1e-14.
    badvalue : optional
        Value to return for invalid inputs. Default is None.
    name : str, optional
        Name of the distribution. Default is None.
    longname : str, optional
        Long descriptive name of the distribution. Default is None.
    shapes : str or None, optional
        Shape parameters (if any) for the distribution. Default is None.
    extradoc : str, optional
        Additional documentation string. Default is None.
    seed : int or np.random.Generator, optional
        Random seed for reproducibility. Default is None.
    """

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
        """
        Fit the CDF estimator to interval-censored data using an EM-like algorithm.

        Parameters
        ----------
        intervals : list of tuple of float
            List of observed intervals (left, right) for each sample.
        max_iter : int, default 50
            Maximum number of iterations for the EM procedure.
        tol : float, default 1e-4
            Convergence tolerance for the probability updates.
        verbose : bool, default False
            If True, prints iteration progress.
        verbose_interval : int, default 1
            Interval at which verbose output is printed.

        Returns
        -------
        self : CensoredCDFEstimator
            The fitted estimator with computed CDF probabilities and interpolator.
        """
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
        """
        Evaluate the CDF at given points.

        Parameters
        ----------
        x : array-like or float
            Points at which to evaluate the CDF.

        Returns
        -------
        array-like or float
            Evaluated CDF values at `x`.
        """
        return self._cdf_interpolator(x)

    def support(self):
        """
        Get the support of the distribution.

        Returns
        -------
        tuple of float
            Lower and upper bounds of the support.
        """
        return self.support_

    @staticmethod
    def _if_sub_interval(interval1: Tuple[float, float], interval2: Tuple[float, float]):
        """
        Check if interval2 is fully contained within interval1.

        Parameters
        ----------
        interval1 : tuple of float
            Parent interval (left, right).
        interval2 : tuple of float
            Candidate sub-interval (left, right).

        Returns
        -------
        bool
            True if interval2 is contained within interval1, False otherwise.
        """
        return (interval1[0] <= interval2[0]) and (interval1[1] >= interval2[1])

    @staticmethod
    def _extract_disjoint_intervals(intervals: List[Tuple[float, float]]):
        """
        Convert overlapping intervals into a sorted array of disjoint intervals.

        Parameters
        ----------
        intervals : list of tuple of float
            List of input intervals (left, right).

        Returns
        -------
        np.ndarray, shape (n_disjoint, 2)
            Array of disjoint intervals covering the same range as the input.
        """
        assert len(intervals) > 0, "At least one interval should be provided"
        assert all(interval[0] <= interval[1] for interval in intervals)
        lefts, rights = zip(*intervals)
        sort_endpoints = sorted([(left, "L") for left in lefts] + [(right, "R") for right in rights])
        disjoint_intervals = []
        for (ep, next_ep) in zip(sort_endpoints[:-1], sort_endpoints[1:]):
            if ep[1] == "L" and next_ep[1] == "R":
                disjoint_intervals.append((ep[0], next_ep[0]))
        return np.array(disjoint_intervals)
