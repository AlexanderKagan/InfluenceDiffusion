
# InfluenceDiffusion

InfluenceDiffusion is a Python library that provides instruments for working with influence diffusion models on graphs. In particular, it contains implementations of
- Popular diffusion models such as Independent Cascade, (General) Linear Threshold, etc.
- Methods for estimating parameters of these models and constructing the corresponding confidence intervals.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install InfluenceDiffusion.

```bash
pip install InfluenceDiffusion
```

## Usage

```python
# Imports
import matplotlib.pyplot as plt
from networkx import connected_watts_strogatz_graph
from scipy.stats import beta

from InfluenceDiffusion.Graph import Graph # class inheriting from nx.DiGraph
from InfluenceDiffusion.Inference import GLTInferenceModule
from InfluenceDiffusion.influence_models import LTM
from InfluenceDiffusion.estimation_models.OptimEstimation import GLTWeightEstimator
from InfluenceDiffusion.weight_samplers import make_random_weights_with_indeg_constraint
from InfluenceDiffusion.plot_utils import plot_with_conf_intervals


# Sample a connected Watts-Strogatz graph
random_state = 1
g = Graph(connected_watts_strogatz_graph(n=100, k=5, p=0.2, seed=random_state))

# Set ground-truth GLT model edge weights (in-degree of each node is at most 1)
weights = make_random_weights_with_indeg_constraint(g, indeg_ub=1, random_state=random_state)
g.set_weights(weights)

# Sample traces from the Beta(2, 1)-GLT model on this graph
threhsold_distrib = beta(2, 1)
gltm = LTM(g, threshold_generator=threhsold_distrib, random_state=random_state)
traces = gltm.sample_traces(1000)

# Estimate the weights using the traces
gltm_estimator = GLTWeightEstimator(g, threhsold_distrib)
pred_weights = gltm_estimator.fit(traces)

# Compute 95% confidence intervals
glt_inferencer = GLTInferenceModule(gltm_estimator)
conf_ints = glt_inferencer.compute_all_weight_conf_ints(alpha=0.05)

# Compare with the ground-truth weights
plot_with_conf_intervals(weights, pred_weights, conf_ints,
                         xlab="True weights", ylab="Predicted weights")
```
![](images/weight_estimation.jpg)

## License

MIT License

Copyright (c) 2024 Alexander Kagan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
