import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import LinearRegression

from sklearn_meta import ForwardStagewiseMeta
from sklearn_meta import BackfittingMeta

metas = [
    ForwardStagewiseMeta,
    BackfittingMeta,
]


@pytest.mark.parametrize("meta", metas)
def test_check_estimator(meta):

    estimators = [LinearRegression() for _ in range(5)]
    pipeline = meta(base_estimators=estimators)
    assert check_estimator(pipeline) is None, (
        "Estimator does not adhere to the scikit-learn interface and standards."
    )
