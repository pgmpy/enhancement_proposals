## 3. Implement `TabularCPD`, `LinearGaussianCPD`, `FunctionalCPD`

### Proposal Solutions

```py
# pgmpy/parameterization/TabularCPD.py
class TabularCPD(BaseParameter):
    def __init__(
        self,
        variable_card,
        evidence_card=None,
        state_names=None,
        prior_type=None,
        equivalent_sample_size=10,
        pseudo_counts=None
    ):
    ...
    # Fitted attributes (sklearn convention)
    values_: np.ndarray             # (variable_card, prod(evidence_card))
    classes_: np.ndarray            # child state labels
    feature_names_in_: np.ndarray   # parent labels in column order (if known)
    is_fitted_: bool

    @classmethod
    def from_values(
        cls,
        variable_card,
        values,
        evidence_card=None,
        state_names=None,
        parent_order=None,
    ):
        return TabularCPD(...)

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        ...
        return CategoricalDistribution() # (len(X), variable_card)
```

```py
# pgmpy/parameterization/LinearGaussianCPD.py
class LinearGaussianCPD(BaseParameter):
    def __init__(self):
        ...

    beta_: np.ndarray               # length n_parents + 1; beta_[0] = intercept
    std_: float
    feature_names_in_: np.ndarray   # parent labels in beta_[1:] order (if known)

    @classmethod
    def from_values(cls, beta, std, parent_order=None):
        ...
        return LinearGaussianCPD()

    def fit(self, X, y, sample_weight=None):
        ...
        return self

    def predict_proba(self, X):
        ...
        return skpro.distributions.Normal(mu, sigma)

    def sample(self, X, n_samples=1):
        return pd.Series()

    def predict_log_proba(self, X):
        return pd.Series()

```

```py
# pgmpy/parameterization/FunctionalCPD.py
class FunctionalCPD(BaseParameter):
    def __init__(self):
        ...

    @classmethod
    def from_values(cls):
        ...
        return FunctionalCPD()

    def fit(self, X, y):
        ...
        return mu, sigma

```

### Alternative Solutions

- `TabularCPD` inherits from `ClassifierMixin`:
    - only the tag system from `ClassifierMixin` can be used, and I think it would be more appropriate to create a pgmpy-specific tag system rather than relying on scikit-learn’s tag system.

- `log_prob` and `predict_log_proba` and (`log_pdf`, `log_pmf`)
    - scikit-learn’s `predict_log_proba()` is a classifier-specific method, and its role is different from that of `log_prob()`, `log_pdf()`, or `log_pmf()`.
    - Also, since it uses `skpro.distribution`, it seems easier to use `log_pdf` and `log_pmf`.
