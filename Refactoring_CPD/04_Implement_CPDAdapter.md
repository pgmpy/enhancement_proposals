## 4. Implement `SkproAdapter`, `SklearnAdapter`

### Proposal Solutions

- `SkproAdapter` inherits from `_DelegatedProbaRegressor`.
    - This allows `SkproAdapter` to have `BaseParameter`'s tag, while preserving the skpro tags in `self.estimator_`.
- `SklearnAdapter` will be follows the same structure.
- The API follows the sklearn API style.(`fit`, `predict_proba`)
- In the case of `predict_proba`, it returns a skpro's distribution object.
- If we use skpro’s distributions, I think we can make use of `log_pmf()` and `log_pdf()`.

```py
# pgmpy/parameterization/adapter/skpro.py
from pgmpy.base import BaseParameter
from skpro.regression.base._delegate import _DelegatedProbaRegressor
from sklearn.base import clone

class SkproAdapter(_DelegatedProbaRegressor, BaseParameter):
    def __init__(self, estimator):
        self.estimator = estimator
        super().__init__()

    def _fit(self, X, y, C=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X=X, y=y, C=C)
        return self

    def sample(self, X, n_samples=1):
        dist = self.predict_proba(X)
        samples = dist.sample()
        return pd.Series(samples, ...)
    
```

- [skpro/regression/base/_delegate.py](https://github.com/sktime/skpro/blob/main/skpro/regression/base/_delegate.py)

```py
# pgmpy/parameterization/adapter/sklearn.py
from pgmpy.base import BaseParameter
from sklearn.base import clone, is_classifier, is_regressor

class SklearnAdapter(BaseParameter):
    def __init__(self, estimator):
        self.estimator = estimator
        super().__init__()

    def _fit(self, X, y, C=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)             
        return self

    def _predict_proba(self, X):
        if is_regressor(self.estimator_):
            from skpro.distribution import Normal
            mu = self.estimator_.predict(X)
            sigma = 1.0 
            return Normal(mu, sigma)

        elif is_classifier(self.estimator_):
            ...
            return CategoricalDistribution(...)

    def sample(self, X, n_samples=1):
        dist = self.predict_proba(X)
        samples = dist.sample()
        return pd.Series(samples, ...)

```

### Alternative Solutions

- Implement migration `CPDAdapter`:
    - skpro's `predict_proba()` is returning skpro's distribution. But, sklearn's one is not returning skpro's distribution. So, I think Implement `SkproAdapter`, `SklearnAdapter` is better.
    - Also, `skpro` deal with only regression model. but, `sklearn` deal with classifier and regression.
- Implement `predict_log_proba()`:
    - scikit-learn’s `predict_log_proba()` is a classifier-specific method, and its role is different from that of `log_prob()`, `log_pdf()`, or `log_pmf()`.
- inheriting from `BaseProbaRegressor`: For the adapter class, I think using the structure of `_DelegatedProbaRegressor` is better.
- By using skpro’s `DeltaPointRegressor`: sklearn regression models can be made to return a delta distribution via `predict_proba()`. [skpro/regression/delta.py](https://github.com/sktime/skpro/blob/main/skpro/regression/delta.py)
