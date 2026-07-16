import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Callable
from functools import partial
from scipy.stats import norm
from scipy.stats._distn_infrastructure import rv_frozen

from .estimation_models.OptimEstimation import GLTWeightEstimator

__all__ = ["GLTInferenceModule"]


class GLTInferenceModule:
    def __init__(self, estimator: GLTWeightEstimator, vertex_2_jax_cdf: Dict[int, Callable] = None):
        self.estimator = estimator
        if vertex_2_jax_cdf is None:
            self.vertex_2_jax_cdf = {vertex: self._make_jax_cdf(distrib)
                                     for vertex, distrib in self.estimator.vertex_2_distrib.items()}
        else:
            self.vertex_2_jax_cdf = vertex_2_jax_cdf
        self.vertex_2_parent_weight_cov = {}

    def _get_vertex_activ_status_and_masks(self, vertex: int):
        n_parents = self.estimator.graph.get_indegree(vertex)
        if vertex in self.estimator._vertex_2_active_parent_mask_t:
            t_active_masks = self.estimator._vertex_2_active_parent_mask_t[vertex]
            tm1_active_masks = self.estimator._vertex_2_active_parent_mask_tm1[vertex]
        else:
            t_active_masks = jnp.empty(shape=(0, n_parents))
            tm1_active_masks = jnp.empty(shape=(0, n_parents))

        if vertex in self.estimator._failed_vertices_masks:
            t_failed_masks = self.estimator._failed_vertices_masks[vertex]
            tm1_failed_masks = jnp.zeros_like(t_failed_masks)
        else:
            t_failed_masks = jnp.empty(shape=(0, n_parents))
            tm1_failed_masks = jnp.empty(shape=(0, n_parents))
        ys = jnp.array([True] * len(t_active_masks) + [False] * len(t_failed_masks), dtype=bool)
        masks_t = jnp.vstack([t_active_masks, t_failed_masks], dtype=jnp.float32)
        masks_tm1 = jnp.vstack([tm1_active_masks, tm1_failed_masks], dtype=jnp.float32)
        return ys, masks_t, masks_tm1

    @staticmethod
    def _make_jax_cdf(distrib: rv_frozen):
        name = distrib.dist.name
        args = distrib.args
        kwargs = distrib.kwds
        local_dic = {}
        exec(f"jax_distrib=jax.scipy.stats.{name}", None, local_dic)
        jax_distrib = local_dic["jax_distrib"]
        if name == "expon":
            return lambda x: 1. - jnp.exp(-x)
        return lambda x: jax_distrib.cdf(x, *args, **kwargs)

    @staticmethod
    def _vertex_ll(parent_weights, cdf, ys, masks_t, masks_tm1, eps=1e-6):
        F_t = cdf(masks_t @ parent_weights)
        F_tm1 = cdf(masks_tm1 @ parent_weights)
        F_tm1 = jnp.where(F_tm1 < eps, 0, F_tm1)
        F_t = jnp.where(F_t > 1. - eps, 1., F_t)
        prob = jnp.clip(F_t - F_tm1, eps, 1. - eps)
        return jnp.sum(jnp.log(jnp.where(ys, prob, 1. - prob)))

    @staticmethod
    def _vertex_conditional_activ_prob(weights, cdf, mask_t, mask_tm1, eps=1e-6):
        F_t = cdf(mask_t @ weights)
        F_tm1 = cdf(mask_tm1 @ weights)
        F_t = jnp.clip(F_t, eps, 1. - eps)
        F_tm1 = jnp.clip(F_tm1, eps, 1. - eps)
        return 1. - (1. - F_t) / (1. - F_tm1)

    def compute_vertex_conditional_activ_probs(self, vertex: int, masks_t, masks_tm1):
        cdf = self.vertex_2_jax_cdf[vertex]
        parent_mask = self.estimator.graph.get_parents_mask(vertex)
        weights = jnp.array(self.estimator.weights_[parent_mask], dtype=jnp.float32)
        return self._vertex_conditional_activ_prob(weights=weights, cdf=cdf, mask_t=masks_t, mask_tm1=masks_tm1)

    def compute_vertex_parent_weight_cov(self, vertex, fisher_eps=1e-6):
        if vertex in self.vertex_2_parent_weight_cov:
            return self.vertex_2_parent_weight_cov[vertex]
        cdf = self.vertex_2_jax_cdf[vertex]
        parent_mask = self.estimator.graph.get_parents_mask(vertex)
        weights = jnp.array(self.estimator.weights_[parent_mask], dtype=jnp.float32)
        ys, masks_t, masks_tm1 = self._get_vertex_activ_status_and_masks(vertex=vertex)
        hessian_fun = jax.hessian(partial(self._vertex_ll, cdf=cdf, ys=ys,
                                          masks_t=masks_t, masks_tm1=masks_tm1))
        fisher_info = -hessian_fun(weights)
        cov = jnp.linalg.inv(fisher_info + fisher_eps * jnp.eye(len(fisher_info)))
        self.vertex_2_parent_weight_cov[vertex] = cov
        return cov

    def compute_vertex_2_parent_weight_cov_dict(self):
        return {vertex: self.compute_vertex_parent_weight_cov(vertex)
                for vertex in self.estimator.informative_vertices}

    def compute_parent_weight_conf_ints(self, vertex: int, alpha: float = 0.05):
        weights = self.estimator.weights_[self.estimator.graph.get_parents_mask(vertex)]
        lb, ub = self.estimator.vertex_2_distrib[vertex].support()
        cov = self.compute_vertex_parent_weight_cov(vertex=vertex)
        stds = jnp.sqrt(jnp.diag(cov))
        quantile = norm.ppf(1 - alpha / 2)
        return jnp.stack([jnp.clip(weights - quantile * stds, lb, None),
                          jnp.clip(weights + quantile * stds, None, ub)]).T

    def compute_all_weight_conf_ints(self, alpha=0.05):
        conf_ints = np.empty(shape=(self.estimator.graph.count_edges(), 2), dtype=np.float32)
        for vertex in self.estimator.informative_vertices:
            parent_conf_ints = self.compute_parent_weight_conf_ints(vertex=vertex, alpha=alpha)
            mask = self.estimator.graph.get_parents_mask(vertex)
            conf_ints[mask] = parent_conf_ints
        return conf_ints

    def compute_vertex_activation_prob_conf_ints(self, vertex: int, masks_t, masks_tm1, alpha=0.05):
        cdf = self.vertex_2_jax_cdf[vertex]
        weights = self.estimator.weights_[self.estimator.graph.get_parents_mask(vertex)]
        cov = self.compute_vertex_parent_weight_cov(vertex=vertex)
        quantile = norm.ppf(1 - alpha / 2)
        conf_ints = []
        for mask_t, mask_tm1 in zip(masks_t, masks_tm1):
            prob, prob_grad = jax.value_and_grad(partial(self._vertex_conditional_activ_prob,
                                                 cdf=cdf, mask_t=mask_t, mask_tm1=mask_tm1))(weights)
            prob_grad = prob_grad.reshape((-1, 1))
            std = jnp.sqrt(prob_grad.T @ cov @ prob_grad)
            conf_ints.append([jnp.clip(prob - quantile * std, 0., None).item(),
                              jnp.clip(prob + quantile * std, None, 1.).item()])

        return np.stack(conf_ints)
