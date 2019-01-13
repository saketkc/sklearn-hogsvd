"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`sklhogsvd.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from sklhogsvd import TemplateEstimator

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
estimator = TemplateEstimator()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
