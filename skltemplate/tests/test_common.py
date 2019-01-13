import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklhogsvd import TemplateEstimator
from sklhogsvd import TemplateClassifier
from sklhogsvd import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
