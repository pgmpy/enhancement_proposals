## 05. Potential ideas

### 05-01. Unsupervised learning models in `sklearn` can serve as the distribution at the root node.

- A representative example is `GaussianMixture`, which is an unsupervised learning model.
- Therefore, it can represent a probability distribution rather than a conditional probability distribution.
- However, to clearly define the scope of the project, I think it would be better to focus on classifiers and regression models.

- [sklearn/mixture/_gaussian_mixture.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_gaussian_mixture.py)

## 05-02. Implement `check_parameterization()`

```py
# pgmpy/utils/parameter_checks.py

def check_parameterization():
    ...

```

- Ref: [skpro/utils/estimator_checks.py](https://github.com/sktime/skpro/blob/main/skpro/utils/estimator_checks.py)
- Ref: [sklearn/utils/estimator_checks.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/estimator_checks.py)
- Ref: [sktime/tests/test_all_estimators.py](https://github.com/sktime/sktime/blob/main/sktime/tests/test_all_estimators.py)
- Ref: [skpro/tests/test_all_estimators.py](https://github.com/sktime/skpro/blob/main/skpro/tests/test_all_estimators.py)
