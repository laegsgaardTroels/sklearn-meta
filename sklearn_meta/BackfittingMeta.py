from sklearn_meta import BaseMeta

import numpy as np

from sklearn.exceptions import NotFittedError


class BackfittingMeta(BaseMeta):

    def __init__(self, base_estimators, n_iter=10):
        self.n_iter = n_iter
        self._alpha = None
        self._mean_centering = [0] * len(base_estimators)
        super().__init__(base_estimators)
        if len(base_estimators) <= 1:
            raise ValueError("Does not make sense with 1 estimator.")

    def _fit(self, X, y):
        self._alpha = np.mean(y)
        for _ in range(self.n_iter):
            for j, estimator in enumerate(self._estimators):
                self._backfit(X, y, j)
                self._mean_center(X, j)
        return self

    def _predict(self, X):
        return self.partial_predict(X, j=None)

    def _backfit(self, X, y, j):
        residual = y - self.partial_predict(X, j)
        self._estimators[j] = self._estimators[j].fit(X, residual)

    def _mean_center(self, X, j):
        estimator = self._estimators[j]
        self._mean_centering[j] = - np.mean(estimator.predict(X))

    def partial_predict(self, X, j):
        prediction = np.ones(X.shape[0]) * self._alpha
        for i, estimator in enumerate(self._estimators):
            if i != j:
                try:
                    new_prediction = estimator.predict(X) * 0.1
                except NotFittedError as e:
                    continue
                prediction = (
                  prediction +
                  new_prediction +
                  self._mean_centering[i]
                )
        return prediction
