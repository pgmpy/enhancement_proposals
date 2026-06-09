## 2. Implement `CategoricalDistribution`

### Proposal Solutions

- Implement `CategoricalDistribution` class.
- This class follow skpro's distribution templates and tag system.

```py
# pgmpy/distribution/categorical.py
from skpro.distributions.base import BaseDistribution

class CategoricalDistribution(BaseDistribution):
    _tags = {
        # packaging info
        # --------------
        "python_version": None,
        # estimator tags
        # --------------
        "python_dependencies": [],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "log_pmf", "pmf", "cdf"],
        "broadcast_init": "on",
    }

    def _log_pmf(self, y):
        ...
        return np.ndarray # 2D

```

- [extension_templates/distributions.py](https://github.com/sktime/skpro/blob/main/extension_templates/distributions.py)

### Alternative Solutions
- The `distribution` class has sklearn-style tags:
    - we designed the `distribution` object to follow the skpro structure.
- Implement this class directly in skpro lib. (https://github.com/sktime/skpro/issues/1003)
