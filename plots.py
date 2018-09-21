from matplotlib import pyplot as plt
from sklearn import linear_model, preprocessing
import numpy as np

from wutils.np import rolling_mean


def add_best_fit_curve(x, y, degree, fit_intercept, **kwargs):
    mm = make_polynomial_mm(x=x, degree=degree)
    model = linear_model.LinearRegression(
            n_jobs=-1, fit_intercept=fit_intercept)
    model.fit(X=mm, y=y)
    predictions = model.predict(X=mm)

    ix = np.argsort(x)
    label = "Best Fit Degree {}".format(degree)
    plt.plot(x[ix], predictions[ix], "--", label=label, **kwargs)


def plot_best_fit(x, y, best_fit_degrees, fit_intercept):
    if not isinstance(best_fit_degrees, list):
        best_fit_degrees = [best_fit_degrees]
    for degree in best_fit_degrees:
        add_best_fit_curve(x, y, degree, fit_intercept)


def plot_rolling_mean(x, window, **matplotlib_kwargs):
    rolling_means = rolling_mean(x, window)
    plt.plot(rolling_means, **matplotlib_kwargs)


def make_polynomial_mm(x, degree):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    poly = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X=x)
