from sklearn_meta import BaseMeta


class ForwardStagewiseMeta(BaseMeta):

    def _fit(self, X, y):
        """Fit each estimator to data using FSAM.

        :param X: Input features.
        :param y: Responses.
        """
        residual = y.copy()
        for idx, estimator in enumerate(self._estimators):
            self._estimators[idx] = estimator.fit(X, residual)
            f_hat = self._estimators[idx].predict(X)
            residual = residual - f_hat
        return self

    def _predict(self, X):
        """Predict using the fitted estimators by summing.

        :param X: Input features.
        """

        prediction = None
        for estimator in self._estimators:
            if prediction is None:
                prediction = estimator.predict(X)
            else:
                prediction = prediction + estimator.predict(X)

        return prediction
