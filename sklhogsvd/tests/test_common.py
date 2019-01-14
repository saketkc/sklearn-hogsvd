import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklhogsvd import HigherOrderGSVD


@pytest.mark.parametrize(
    "Estimator", [HigherOrderGSVD]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
