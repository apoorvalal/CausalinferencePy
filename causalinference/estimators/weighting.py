from __future__ import division
import numpy as np

from .base import Estimator
from .ols import calc_cov, calc_ate, calc_ate_se


class Weighting(Estimator):

    """
    Dictionary-like class containing treatment effect estimates.
    """

    def __init__(
        self,
        data,
        estimand="ate",
        use_covariates=False,
        final_est="dim",
        hajekize=False,
    ):
        self._method = "Weighting"
        Y, D, X = data["Y"], data["D"], data["X"]
        pscore = data["pscore"]
        weights = calc_weights(pscore, D, estimand, hajekize)
        if final_est == "reg":
            Y_w, Z_w = weigh_data(Y, D, X, weights, use_covariates)
            wlscoef = np.linalg.lstsq(Z_w, Y_w)[0]
            u_w = Y_w - Z_w.dot(wlscoef)
            cov = calc_cov(Z_w, u_w)
            self._dict = dict()
            self._dict["ate"] = calc_ate(wlscoef)
            self._dict["ate_se"] = calc_ate_se(cov)
        elif final_est == "dim":
            weights[D == 0] = -1 * weights[D == 0]
            self._dict = dict()
            self._dict["ate"] = (weights * Y).mean()
            self._dict["ate_se"] = np.nan


def calc_weights(pscore, D, estimand, hajekize=False):
    N = pscore.shape[0]
    weights = np.empty(N)
    if estimand == "ate":
        weights[D == 0] = 1 / (1 - pscore[D == 0])
        weights[D == 1] = 1 / pscore[D == 1]
    elif estimand == "att":
        weights[D == 1] = 1 / pscore[D == 1]
        weights[D == 0] = pscore[D == 0] / (1 - pscore[D == 0])
    elif estimand == "atc":
        weights[D == 0] = 1 / (1 - pscore[D == 0])
        weights[D == 1] = pscore[D == 1] / (1 - pscore[D == 1])
    elif estimand == "ato":
        weights[D == 0] = pscore[D == 0]
        weights[D == 1] = 1 - pscore[D == 1]
    if hajekize:
        weights[D == 0] = weights[D == 0] / weights[D == 0].mean()
        weights[D == 1] = weights[D == 1] / weights[D == 1].mean()
    return weights


def weigh_data(Y, D, X, weights, use_covariates=False):
    N, K = X.shape
    Y_w = weights * Y
    if use_covariates:
        Z_w = np.empty((N, K + 2))
        Z_w[:, 2:] = weights[:, None] * X
    else:
        Z_w = np.empty((N, 2))
    Z_w[:, 0] = weights
    Z_w[:, 1] = weights * D
    return (Y_w, Z_w)
