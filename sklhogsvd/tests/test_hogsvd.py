import pytest
import numpy as np

from sklearn.utils.testing import assert_allclose

from sklhogsvd import HigherOrderGSVD



def test_template_transformer_error():
    np.random.seed(42)
    data1 = np.random.normal(loc=0, scale=10, size=(10, 10))
    data1 = np.random.normal(loc=0, scale=10, size=(10, 10))
    data2 = np.random.normal(loc=0, scale=10, size=(10, 10))
    data3 = np.random.normal(loc=0, scale=10, size=(10, 10))
    X =  [data1, data2, data3]
    fit = HigherOrderGSVD().fit(X)
    assert_allclose(data1, np.dot(np.dot(fit.U[0,:,:], np.diag(fit.sigmas[0])), fit.V.T))


