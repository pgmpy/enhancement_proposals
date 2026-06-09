## 1. Implement `BaseParameter`

### Proposal Solutions

- scikit-learn and skpro each define tag lists tailored to their own libraries.
- Rather than aligning directly with either library’s tag set, it may be better to follow the tagging-system pattern while defining a separate set of tags specifically for pgmpy.
- This follow `sklearn`'s tagging system.
- rename exist path `pgmpy/base/` -> `pgmpy/graph/`
- create path `pgmpy/base/`. This path has a responsibility of `BaseEstimator` and `Mixin` class.

> In scikit-learn, tags are used to dynamically run tests based on estimator properties, such as input data validation and automated common unit tests. <br>
> In skpro, tags provide users with a way to search for models that can perform a specific workflow, for example through `all_objects()`, `all_tags()`. <br>
> In pgmpy, a model’s tags are used to determine whether a specific algorithm can be run.

```py
from skbase.base import BaseEstimator as _BaseEstimator

class _ParameterTags:
    _config = {}

    _tags = {
        "variable_type" :"discrete"
        "produces_factor" : False
        "is_linear_gaussian": False
        "supports_fit_joint": False
        "python_dependencies" = []
    }

class BaseParameter(_ParameterTags, _BaseEstimator):
    ...
```

- [skpro/base/_base.py](https://github.com/sktime/skpro/blob/main/skpro/base/_base.py)


### Alternative Solutions

- rename exist path `pgmpy/base/` -> `pgmpy/graph/`
- create path `pgmpy/base/`. This path has a resopnsibity of `BaseEstimator` and Tagging system.


```py
# pgmpy/utils/_tags.py
@dataclass(slots=True)
class Tags:
    accessor_type: str | None
    parameter_tags: ParameterTags | None = None

@dataclass(slots=True)
class ParameterTags:
    variable_type: str | None # "discrete" or "continous"
    produces_factor: bool = False
    is_linear_gaussian: bool = False
    supports_fit_joint: bool = False
    distribution: bool = False # Is this CPD can be root node's distribution? (P(X))
    python_dependencies = []

```

```py
# pgmpy/base/_base.py
from sklearn.base import BaseEstimator as _BaseEstimator

class BaseParameter(_BaseEstimator):
    def __pgmpy_tags__(self):
        return Tags(
            accessor_type="parameter",
            parameter_tags=ParameterTags(),
        )

```

- [sklearn/base.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py)
- [sklearn/utils/_tags.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_tags.py)
