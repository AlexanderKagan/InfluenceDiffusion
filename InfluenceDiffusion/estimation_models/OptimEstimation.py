import numpy as np
from warnings import warn
from typing import List, Union, Dict, Tuple, Optional
from scipy.optimize import LinearConstraint, minimize, OptimizeResult
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats import rv_continuous, beta
from copy import copy
from collections import defaultdict

from ..Graph import Graph
from ..Trace import Traces, PseudoTraces
from .BaseWeightEstimator import BaseGLTWEightEstimator, BaseWeightEstimator

__all__ = ["GLTGridSearchEstimator", "GLTWeightEstimator", "GLTWeightDistribEstimator"]


class GLTWeightEstimator(BaseGLTWEightEstimator):

    def _compute_failed_vertex_ll(self, vertex: int, 
                                  parent_weights: np.ndarray, distrib: rv_frozen, eps: float = 1e-8) -> float:
        """Compute the log-likelihood for failed vertices.

        Parameters
        ----------
        vertex : int
            The vertex index.
        parent_weights : np.ndarray
            The weights for the vertex parent edges.
        distrib : rv_frozen
            The probability distribution for the vertex threshold.
        eps : float, optional
            The epsilon value for clipping the failure probabilities.

        Returns
        -------
        float
            The log-likelihood for the failed vertex.
        """
        if vertex not in self._failed_vertices_masks:
            return 0.0
        failed_vertices_indeg = self._failed_vertices_masks[vertex] @ parent_weights
        failure_probs = 1.0 - distrib.cdf(failed_vertices_indeg)
        ll = np.log(np.clip(failure_probs, eps, None)).sum()
        return ll

    def _compute_activated_vertex_ll(self, vertex: int, 
                                     parent_weights: np.ndarray, distrib: rv_frozen, eps: float = 1e-8) -> float:
        """Compute the log-likelihood for activated vertices.

        Parameters
        ----------
        vertex : int
            The vertex index.
        parent_weights : np.ndarray
            The weights for the vertex parent edges.
        distrib : rv_frozen
            The probability distribution for the vertex threshold.
        eps : float, optional
            The epsilon value for clipping the activation probabilities.

        Returns
        -------
        float
            The log-likelihood for the activated vertex.
        """
        if vertex not in self._vertex_2_active_parent_mask_tm1:
            return 0.0
        activated_vertices_indeg_tm1 = self._vertex_2_active_parent_mask_tm1[vertex] @ parent_weights
        activated_vertices_indeg_t = self._vertex_2_active_parent_mask_t[vertex] @ parent_weights
        activation_probs = distrib.cdf(activated_vertices_indeg_t) - distrib.cdf(activated_vertices_indeg_tm1)
        ll = np.log(np.clip(activation_probs, eps, None)).sum()
        return ll

    def _compute_vertex_nll(self, vertex: int, 
                            weights: np.ndarray, distrib: rv_frozen, eps: float = 1e-8) -> float:
        """Compute the negative log-likelihood for a vertex.

        Parameters
        ----------
        vertex : int
            The vertex index.
        weights : np.ndarray
            The weights for the vertex parent edges.
        distrib : rv_frozen
            The probability distribution for the vertex threshold.
        eps : float, optional
            The epsilon value for clipping probabilities.

        Returns
        -------
        float
            The negative log-likelihood for the vertex.
        """
        ll = self._compute_activated_vertex_ll(vertex, weights, distrib, eps) + \
             self._compute_failed_vertex_ll(vertex, weights, distrib, eps)
        return -ll
    
    def _compute_normalized_vertex_nll(self, vertex: int, 
                                       weights: np.ndarray, distrib: rv_frozen, eps: float = 1e-8) -> float:
        """Compute the normalized negative log-likelihood for a vertex.

        Parameters
        ----------
        vertex : int
            The vertex index.
        weights : np.ndarray
            The weights for the vertex parent edges.
        distrib : rv_frozen
            The probability distribution for the vertex threshold.

        Returns
        -------
        float
            The normalized negative log-likelihood for the vertex.
        """
        num_informative_traces = self._vertex_2_num_traces_was_informative.get(vertex, 1)
        return self._compute_vertex_nll(vertex, weights=weights, distrib=distrib, eps=eps) / num_informative_traces

    def _compute_total_nll(self, weights: np.ndarray, 
                           vertex_2_distrib: Optional[Dict[int, rv_frozen]] = None, 
                           eps: float = 1e-8) -> float:
        """Compute the total negative log-likelihood for informative vertices.

        Parameters
        ----------
        weights : np.ndarray
            The weights for the graph edges.
        vertex_2_distrib : Dict[int, rv_frozen], optional
            Dictionary mapping vertices to the probability distributions of their thresholds.
            If None, the estimator's current vertex_2_distrib is used.

        Returns
        -------
        float
            The total negative log-likelihood.
        """
        if vertex_2_distrib is None:
            vertex_2_distrib = self.vertex_2_distrib

        return np.sum([
            self._compute_vertex_nll(vertex, 
                                     weights=weights[self.graph.get_parents_mask(vertex)], 
                                     distrib=vertex_2_distrib[vertex], 
                                     eps=eps)
            for vertex in self.informative_vertices]
        )

    def _make_parent_weight_constraints(self, vertex: int, distrib: rv_frozen, eps: float = 1e-8) -> LinearConstraint:
        """Create constraints for parent weights of a vertex.

        Parameters
        ----------
        vertex : int
            The vertex index.
        distrib : rv_frozen
            The probability distribution for the vertex threshold.
        eps : float, optional
            Small value for numerical stability.

        Returns
        -------
        LinearConstraint
            The linear constraints for optimization.
        """
        indeg_max = distrib.support()[1]
        num_parents = int(self.graph.get_indegree(vertex, weighted=False))
        mat_constrain = np.vstack([np.eye(num_parents), np.ones(num_parents)])
        lb = eps * np.ones(num_parents + 1)
        ub = (indeg_max - eps) * np.ones(num_parents + 1)
        return LinearConstraint(A=mat_constrain, lb=lb, ub=ub)

    def _precompute_all_constraints(self) -> None:
        """Precompute weight constraints for all informative vertices."""
        self._weight_constraints = {
            vertex: self._make_parent_weight_constraints(vertex, self.vertex_2_distrib[vertex])
            for vertex in self.informative_vertices
        }
    
    def _optimize_vertex_parent_weights(self, vertex: int, distrib: rv_frozen, x0: Optional[np.ndarray] = None
                                        ) -> OptimizeResult:
        """Optimize parent weights for a specific vertex.

        Parameters
        ----------
        vertex : int
            The vertex to optimize.
        distrib : rv_frozen
            The probability distribution for the vertex threshold.
        x0 : np.ndarray, optional
            Initial guess for the parent weights.

        Returns
        -------
        OptimizeResult
            The result of the optimization.
        """
        x0 = self.weights_[self.graph.get_parents_mask(vertex)] if x0 is None else x0
        optimizer_output = minimize(
            lambda weights: self._compute_normalized_vertex_nll(vertex=vertex, weights=weights, distrib=distrib),
            x0=x0,
            method='SLSQP',
            constraints=self._weight_constraints[vertex]
        )
        return optimizer_output

    def _fit_vertex_parent_params(self, vertex: int) -> Tuple[np.ndarray, rv_frozen]:
        """Fit parameters for a specific vertex.

        Parameters
        ----------
        vertex : int
            The vertex to optimize.

        Returns
        -------
        Tuple[np.ndarray, rv_frozen]
            Fitted weights and threshold distribution for the vertex.
        """
        distrib = self.vertex_2_distrib[vertex]
        optimizer_output = self._optimize_vertex_parent_weights(vertex, distrib=distrib)
        return optimizer_output.x, distrib

    def _set_informative_vertices_parent_params(self, informative_vertices_params: List[Tuple[np.ndarray, rv_frozen]]) \
            -> None:
        """Set the optimized parameters for informative vertices.

        Parameters
        ----------
        informative_vertices_params : List[Tuple[np.ndarray, rv_frozen]]
            List of optimized parameters for each informative vertex.
        """
        for vertex, (parent_weights, vertex_distrib) in zip(self.informative_vertices, informative_vertices_params):
            self.weights_[self.graph.get_parents_mask(vertex)] = parent_weights
            self.vertex_2_distrib[vertex] = vertex_distrib

    def _pre_fit(self, traces: Union[Traces, PseudoTraces], 
                 init_weights: Optional[Union[List[float], np.ndarray]] = None, 
                 masks_path: Optional[str] = None) -> None:
        """Prepare for fitting by preprocessing traces and initializing constraints.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to analyze.
        init_weights : Optional[Union[List[float], np.ndarray]]
            Initial weights to set.
        masks_path : Optional[str]
            Path to load masks from.
        """
        self._preprocess_traces(traces, masks_path=masks_path)
        self._precompute_all_constraints()
        self._init_weights(init_weights)


class GLTGridSearchEstimator(GLTWeightEstimator):
    """GLT Grid Search Estimator for hyperparameter tuning.

    Attributes
    ----------
    vertex_2_distrib_grid : Dict[int, List[rv_frozen]]
        Distribution grid for each vertex.
    """

    def __init__(self, graph: Graph,
                 distribs_grid: Union[Dict[int, List[rv_frozen]], List[rv_frozen]],
                 n_jobs: int = 1) -> None:
        """Initialize the GLT Grid Search Estimator.

        Parameters
        ----------
        graph : Graph
            The graph structure.
        distribs_grid : Union[Dict[int, List[rv_frozen]], List[rv_frozen]]
            The grid of distributions for each vertex.
        n_jobs : int, optional
            Number of jobs for parallel processing.
        """
        super().__init__(graph=graph, n_jobs=n_jobs)
        self._validate_grid(distribs_grid)

    def _validate_grid(self, distribs_grid: Union[Dict[int, List[rv_frozen]], List[rv_frozen]]) -> None:
        """Validate the distribution grid.

        Parameters
        ----------
        distribs_grid : Union[Dict[int, List[rv_frozen]], List[rv_frozen]]
            The grid of distributions to validate.

        Raises
        ------
        AssertionError
            If the distribution grid does not match the graph vertices.
        """
        if isinstance(distribs_grid, (List, Tuple)):
            self.vertex_2_distrib_grid = {vertex: distribs_grid for vertex in self.graph.get_vertices()}
        elif isinstance(distribs_grid, Dict):
            assert set(distribs_grid.keys()) == self.graph.get_vertices(), \
                "If `distribs_grid` is a dict, its keys should be the vertices of the graph."
            self.vertex_2_distrib_grid = distribs_grid
        else:
            raise NotImplementedError("`distribs_grid` should be either a list, tuple, or a dict.")

        assert all([
            all([isinstance(distrib, rv_frozen) for distrib in vertex_distribs])
            for vertex_distribs in self.vertex_2_distrib_grid.values()
        ]), "All elements of the grid should be `rv_frozen`."

        assert all([
            all([distrib.support() == vertex_distribs[0].support() for distrib in vertex_distribs])
            for vertex_distribs in self.vertex_2_distrib_grid.values()
        ]), "All distributions for each vertex should have the same support."

    def _optimize_vertex_parent_params(self, vertex: int) -> \
            Tuple[np.ndarray, rv_frozen]:
        """Optimize parameters for a specific vertex using grid search.

        Parameters
        ----------
        vertex : int
            The vertex to optimize.

        Returns
        -------
        Tuple[np.ndarray, rv_frozen]
            Fitted weights and threshold distribution for the vertex.
        """
        best_nll = np.inf
        for distrib in self.vertex_2_distrib_grid[vertex]:
            optimizer_output = self._optimize_vertex_parent_weights(vertex, distrib=distrib)
            nll = optimizer_output.fun
            if nll < best_nll:
                best_weights = optimizer_output.x
                best_distrib = distrib
                best_nll = nll

        return best_weights, best_distrib


class GLTWeightDistribEstimator(GLTWeightEstimator):
    """
    Estimator of weights and vertex threshold distributions under the GLT diffusion model.
    This estimator alternates estimation of the GLT weights and estimation of the vertex threshold parameters.
    """
    def __init__(self, graph: Graph,
                 distrib_family: rv_continuous = beta, 
                 distrib_params_2_range: Dict[str, Tuple[float, float]] = {"a": (1, 10), "b": (1, 10)},
                 fixed_distrib_params: Optional[Dict[str, float]] = None,
                 n_jobs: int = 1) -> None:

        """Initialize the GLTWeightDistribEstimator.

        Parameters
        ----------
        graph : Graph
            The graph structure.
        distrib_family : rv_continuous
            The family of distributions for the vertex thresholds.
        distrib_params_2_range : Dict[str, Tuple[float, float]]
            The ranges for the parameters of the distribution to fit.
        fixed_distrib_params : Dict[str, float], optional
            Fixed parameters for the distribution.
        n_jobs : int, optional
            Number of jobs for parallel processing.
        """
        self.distrib_family = distrib_family
        self.fixed_distrib_params = fixed_distrib_params
        self._validate_distrib_param_ranges(distrib_params_2_range)
        super().__init__(graph=graph, vertex_2_distrib=None, n_jobs=n_jobs)

    def _validate_distrib_param_ranges(self, distrib_params_2_range) -> None:
        """Validate the distribution parameter ranges.

        Parameters
        ----------
        distrib_params_2_range : Dict[str, Tuple[float, float]]
            The ranges for the parameters of the distribution to fit.
        """
        assert all(len(rng) == 2 for rng in distrib_params_2_range.values()), \
            "Each value in distrib_params_2_range should be a tuple of (low, high)."
        assert all(low < high for low, high in distrib_params_2_range.values()), \
            "For each parameter, low should be less than high."
        self.distrib_params_2_range = distrib_params_2_range
          
    def _validate_vertex_distributions(self, vertex_2_distrib: Optional[Union[Dict[int, rv_frozen], rv_frozen]] = None
                                       )-> None:
        """Initialize vertex distributions using the distribution family and base parameters 
        specified during initialization.
        """
        self.distrib_param_names = list(self.distrib_params_2_range.keys())
        self.base_distrib_params = tuple((low + high) / 2 for low, high in self.distrib_params_2_range.values())
        self.vertex_2_distrib = {vertex: self._make_distrib(self.base_distrib_params) 
                                 for vertex in self.graph.get_vertices()}
    
    def _init_history(self) -> None:
        """Initialize history dictionaries for weights, distribution parameters, and negative log-likelihoods."""
        self.weights_history_ = defaultdict(list)
        self.distrib_param_history_ = defaultdict(list)
        self.nll_history_ = defaultdict(list)

    def _update_history(self, vertex: int, weights: np.ndarray, distrib_params: np.ndarray, nll: float) -> None:
        """Update history dictionaries with the latest weights, distribution parameters, and negative log-likelihood.

        Parameters
        ----------
        vertex : int
            The vertex index.
        weights : np.ndarray
            The weights for the vertex parent edges.
        distrib_params : np.ndarray
            The parameters for the vertex threshold distribution.
        nll : float
            The negative log-likelihood for the vertex.
        """
        self.weights_history_[vertex].append(weights.copy())
        self.distrib_param_history_[vertex].append(distrib_params.copy())
        self.nll_history_[vertex].append(nll)

    def fit(self, traces: Union[Traces, PseudoTraces], 
        init_weights: Optional[Union[List, np.ndarray]] = None,
        verbose: bool = False,
        masks_path: Optional[str] = None,
        max_alternate_iter: int = 10,
        tol: float = 1e-5) -> "BaseWeightEstimator":

        """Fit the model to the given traces.

        Parameters
        ----------
        traces : Union[Traces, PseudoTraces]
            The traces to analyze.
        init_weights : Optional[Union[List, np.ndarray]]
            Initial weights to set.
        verbose : bool, optional
            If True, progress messages are printed.
        masks_path : Optional[str]
            Path to load masks from.
        max_alternate_iter : int, optional
            Maximum number of alternating iterations for weight and distribution estimation.
        tol : float, optional
            Tolerance for convergence in alternating iterations.

        Returns
        -------
        self : object
            The fitted estimator instance.
        """
        self._init_history()
        return super().fit(traces=traces, init_weights=init_weights, verbose=verbose, masks_path=masks_path,
                           max_alternate_iter=max_alternate_iter, tol=tol)
    
    def _make_distrib(self, distrib_params: tuple[float, ...]) -> rv_frozen:
        """Create a distribution instance from the given parameters.

        Parameters
        ----------
        distrib_params : tuple of floats
            The parameters for the distribution.

        Returns
        -------
        rv_frozen
            The distribution instance.
        """
        assert len(distrib_params) == len(self.distrib_param_names)
        params_dict = dict(zip(self.distrib_param_names, distrib_params))
        if self.fixed_distrib_params:
            params_dict.update(self.fixed_distrib_params)
        return self.distrib_family(**params_dict)

    def _optimize_vertex_threshold_distrib_params(self, vertex: int, parent_weights: np.ndarray,
                                                  x0: Optional[np.ndarray] = None) -> OptimizeResult:
        """Optimize the threshold distribution parameters for a specific vertex.

        Parameters
        ----------
        vertex : int
            The vertex to optimize.
        parent_weights : np.ndarray
            The weights for the vertex parent edges.
        x0 : np.ndarray, optional
            Initial guess for the threshold distribution parameters.

        Returns
        -------
        OptimizeResult
            The optimization result containing the optimized threshold distribution for the vertex.
        """
        x0 = np.array(self.base_distrib_params) if x0 is None else x0
        optimizer_output = minimize(
            lambda distrib_params: self._compute_normalized_vertex_nll(
                vertex=vertex, weights=parent_weights, distrib=self._make_distrib(distrib_params)),
            x0=x0,
            method='SLSQP',
            constraints=self._distrib_param_constraints[vertex]
        )
        return optimizer_output
    
    def _make_distrib_param_constraints(self) -> LinearConstraint:
        """Create constraints for the distribution parameters.

        Returns
        -------
        LinearConstraint
            The linear constraints for optimization.
        """
        mat_constrain = np.eye(len(self.distrib_param_names))
        lb = np.array([self.distrib_params_2_range[param][0] for param in self.distrib_param_names])
        ub = np.array([self.distrib_params_2_range[param][1] for param in self.distrib_param_names])
        return LinearConstraint(A=mat_constrain, lb=lb, ub=ub)

    def _precompute_all_constraints(self) -> None:
        """Precompute weight and distribution parameter constraints for all informative vertices."""
        super()._precompute_all_constraints()
        distib_param_constraint = self._make_distrib_param_constraints()
        self._distrib_param_constraints = {vertex: distib_param_constraint for vertex in self.informative_vertices}

    def _fit_vertex_parent_params(self, vertex: int, max_alternate_iter: int = 10, tol: float = 1e-5
                                  ) -> Tuple[np.ndarray, rv_frozen]:
        """Optimize vertex parametersby iteratively estimating the weights and vertex threshold parameters.

        Parameters
        ----------
        vertex : int
            The vertex to optimize.
        max_alternate_iter: int
            Max number of weight & CDF estimation steps
        tol : float, optional
            Tolerance for convergence in alternating iterations.

        Returns
        -------
        Tuple[np.ndarray, rv_frozen]
            Weights and vertex threshold distribution.
        """

        parent_weights = self.weights_[self.graph.get_parents_mask(vertex)].copy()
        distrib_params = np.array(self.base_distrib_params).copy()
        for _ in range(max_alternate_iter):
            distrib_optim_out = self._optimize_vertex_threshold_distrib_params(vertex, parent_weights=parent_weights,
                                                                               x0=distrib_params)
            distrib = self._make_distrib(distrib_optim_out.x)
            weight_optim_out = self._optimize_vertex_parent_weights(vertex, distrib=distrib, 
                                                                    x0=parent_weights)

            self._update_history(vertex, weight_optim_out.x, distrib_optim_out.x, weight_optim_out.fun)
            
            # print(f"Alternating Iteration: {iteration}, Weight Diff: {diff:.4f}")
            if  np.linalg.norm(weight_optim_out.x - parent_weights) < tol:
                break
            parent_weights = weight_optim_out.x
            distrib_params = distrib_optim_out.x

        return parent_weights, distrib
    
    def get_fitted_threshold_distrib_params(self) -> Dict[int, Dict[str, float]]:
        """Get the fitted threshold distribution parameters for all vertices.

        Returns
        -------
        Dict[int, Dict[str, float]]
            A dictionary mapping each vertex to its fitted threshold distribution parameters.
        """
        return {vertex: self.vertex_2_distrib[vertex].kwds for vertex in self.informative_vertices}
