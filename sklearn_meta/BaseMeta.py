from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone

from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_is_fitted


class BaseMeta(BaseEstimator, RegressorMixin):

    def __init__(self, base_estimators):
        if not isinstance(base_estimators, list):
            raise ValueError(
                "Input type should be list "
                f"not {type(base_estimators)}"
            )
        for base_estimator in base_estimators:
            if isinstance(base_estimator, BaseEstimator):
                continue
            else:
                raise ValueError(f"Invalid input type: {type(estimator)}")
        self.base_estimators = base_estimators

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self._estimators = [
            clone(estimator) for estimator in self.base_estimators
        ]
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        raise NotImplemented("Implement in subclass.")

    def predict(self, X):
        # Check if fit has been called.
        check_is_fitted(self, ['_estimators'])
        return self._predict(X)

    def _predict(self, X, y):
        raise NotImplemented("Implement in subclass.")
