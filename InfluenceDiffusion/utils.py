import numpy as np
from typing import Iterable, Set
from scipy.stats._distn_infrastructure import rv_frozen
import jax.numpy as jnp


def multiple_union(set_list: Iterable[Set]):
    final_set = set()
    for cur_set in set_list:
        final_set = final_set.union(cur_set)
    return final_set


def invert_non_zeros(array):
    out = np.array(array, dtype=float)
    non_zero_mask = array != 0
    out[non_zero_mask] = 1. / out[non_zero_mask]
    return out


def random_vector_inside_simplex(dim: int, ub: float = 1.):
    U = np.random.uniform(low=0, high=ub, size=dim)
    U_sorted = np.sort(U)
    U_sorted = np.concatenate(([0], U_sorted))
    x = np.diff(U_sorted)
    return x


def random_vector_on_simplex(dim: int, ub: float = 1.):
    X = random_vector_inside_simplex(dim=dim, ub=1)
    return X / np.sum(X) * ub


def make_jax_cdf(distrib: rv_frozen):
    name = distrib.dist.name
    args = distrib.args
    kwargs = distrib.kwds
    local_dic = {}
    exec(f"jax_distrib=jax.scipy.stats.{name}", None, local_dic)
    jax_distrib = local_dic["jax_distrib"]
    if name == "expon":
        return lambda x: 1. - jnp.exp(-x)
    return lambda x: jax_distrib.cdf(x, *args, **kwargs)