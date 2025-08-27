from functools import partial
from typing import List

import geoplot as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_validate
from statsmodels.tools.eval_measures import rmse


class FeOi_prediction:
    """
    Class for predicting melt FeO from differentiation trends.

    Calculate multiple linear regressions by ordinary least-squares (OLS) of FeO against melt major element compositions.

    Parameters
    ----------
    x : pandas DataFrame
        melt compositions (excluding FeO) in oxide wt. %. Columns are used as predictors.
    FeO : pandas Series
        melt FeO contents in wt. %

    Attributes
    ----------
    predictors : List[str]
        names of elements used as predictors. Defaults to all columns in ``x``
    coefficients   : pandas Series
        fitted regression coefficients
    errors : pandas Series
        one standard deviation errors on ``coefficients``
    model : Callable
        FeOi prediction model
    """

    def __init__(self, x: pd.DataFrame, FeO: pd.Series):

        self.x = x
        self.FeO = FeO

        self.predictors = x.columns.values

    @property
    def coefficients(self):

        if not hasattr(self, "_slopes"):
            raise AttributeError("No regression found")

        return pd.concat([pd.Series({"intercept": self._intercept}), self._slopes])

    @property
    def errors(self):

        if not hasattr(self, "_slopes_error"):
            raise AttributeError("No regression found")

        return pd.concat(
            [pd.Series({"intercept": self._intercept_error}), self._slopes_error]
        )

    @property
    def model(self) -> callable:
        return partial(self._FeO_initial_func, coefficients=self.coefficients)

    def get_OLS_coefficients(self) -> None:
        """
        Get linear regression coefficients by ordinary least squares.

        Uses elements in ``predictors`` to predict FeO. Results are stored in ``coefficients`` and ``errors``.
        """

        reg_ols = self._OLS_fit()

        self._slopes = reg_ols.params
        self._slopes_error = reg_ols.bse

        self._intercept = self._slopes.pop("const")
        self._intercept_error = self._slopes_error.pop("const")

    def random_sample_coefficients(self, n: int) -> pd.DataFrame:
        """
        Randomly sample fitted slopes within their errors. Intercepts, :math:`\hat{\\beta}_{0}`, are calculated as:

        .. math ::
            \hat{\\beta}_{0} = \overline{y} - \sum \hat{\\beta}_{1}^{e} \cdot \overline{e}

        for elements *e* in ``predictors`` with slopes :math:`\hat{\\beta}_{1}^{e}` and where *y* = FeO.

        Parameters
        ----------
        n : int
            amount of samples

        Returns
        -------
        coefficients : pandas DataFrame
            DataFrame with resampled coefficients.
        """

        if not hasattr(self, "errors"):
            self.get_OLS_coefficients()

        coeff_MC = pd.DataFrame(
            np.random.normal(
                loc=self._slopes, scale=self._slopes_error, size=(n, len(self._slopes))
            ),
            columns=self._slopes.index,
            dtype=np.float16,
        )

        coeff_MC["intercept"] = (
            self.FeO.mean() - coeff_MC.mul(self.x.mean(), axis=1).sum(axis=1)
        ).astype(np.float16)
        # coeff_MC = coeff_MC.squeeze()

        return coeff_MC

    def _OLS_fit(self, x=None, y=None) -> sm.regression.linear_model.RegressionResults:

        if x is None:
            x = self.x[self.predictors]
        if y is None:
            y = self.FeO

        x_train = sm.add_constant(x)

        return sm.OLS(y, x_train).fit()

    def _f_test(
        self,
        data: pd.DataFrame,
        regression1: sm.regression.linear_model.RegressionResults,
        regression2: sm.regression.linear_model.RegressionResults,
    ) -> float:
        """
        F test for models comparison
        """
        return (
            abs((regression1.resid**2).sum() - (regression2.resid**2).sum())
            / (len(regression2.params) - len(regression1.params))
            / ((regression2.resid**2).sum() / (len(data) - len(regression2.params)))
        )

    def _f_to_p(
        self,
        data: pd.DataFrame,
        regression1: sm.regression.linear_model.RegressionResults,
        regression2: sm.regression.linear_model.RegressionResults,
    ) -> float:
        """
        convert F-value to associated P-value
        """
        f = self._f_test(data, regression1, regression2)
        return stats.f.cdf(
            f,
            abs(len(regression1.params) - len(regression2.params)),
            len(data) - len(regression2.params),
        )

    def _compare_models(self, x, y):
        """
        Regress y on x-1 and compare with regressions of y on x
        """
        # Regress y on x
        upper_reg = self._OLS_fit(x, y)

        results = pd.DataFrame(
            columns=["f_ratio", "p_value", "r_ratio", "element_dropped"],
            index=np.arange(0, x.shape[1], 1),
            # dtype=float,
        )
        parameters = {i: None for i in results.index}
        models = {i: None for i in results.index}

        # Regress y on x-1 for all permutations
        for col_drop in results.index:

            x_reduced = x.drop(x.columns[col_drop], axis=1)
            results.loc[col_drop, "element_dropped"] = x.columns[col_drop]

            models[col_drop] = self._OLS_fit(x_reduced, y)
            # Compare with y regressed on x
            results.loc[col_drop, "f_ratio"] = self._f_test(
                x, models[col_drop], upper_reg
            )
            results.loc[col_drop, "p_value"] = self._f_to_p(
                x, models[col_drop], upper_reg
            )
            results.loc[col_drop, "r2"] = models[col_drop].rsquared

            parameters[col_drop] = pd.concat(
                [models[col_drop].params, models[col_drop].pvalues], axis=1
            )
            parameters[col_drop].columns = ["coefficients", "p_value"]

        return results, parameters, models

    def _get_model(self, x, y):
        """
        Compare regression results and select the statistically best model
        """

        # Get models
        results, parameters, models = self._compare_models(x, y)

        # Select the model with the lowest p vaue
        max_index = results["p_value"].idxmin()
        x_data = x.drop(x.columns[max_index], axis=1)
        return (
            x_data,
            parameters[max_index],
            models[max_index],
        )

    def calculate_model_fits(
        self, exclude: List[str] = None, crossvalidation_split: float = 0.15
    ) -> pd.DataFrame:
        """
        Calculate cross-validated misfits of linear regressions.

        *n - 1* regressions are calculated for *n* columns in x. For each new regression, the element whose removal results in the lowest regression F-test p-value is removed from the dataset. Goodness of fit for each regression is measured by R\ :sup:`2`, root-mean squared error (RMSE) and cross-validated RMSE (CV-RMSE), where large differences between RMSE and CV-RMSE (:math:`{\Delta}`\ RMSE) indicate overfitting. As long as RMSE and R\ :sup:`2` values are acceptable, the model with the smallest :math:`{\Delta}`\ RMSE should be selected.

        Parameters
        ----------
        exclude : List[str]
            list of column names in x to exclude from the regression
        crossvalidation_split : float
            split for cross validation. Data fraction (1 - `crossvalidation_split`) will be used for the regression, the rest for calculating misfits by RMSE.

        Returns
        -------
        regressions : pandas DataFrame
            regression results with columns for fitted coefficients (intercept and element-wise slopes), calibration RMSE, cross-validated RMSE, :math:`{\Delta}`\ RMSE and R\ :sup:`2`.
        """

        x = self.x.copy()
        if exclude is not None:
            x = x.drop(columns=exclude)

        y = self.FeO.copy()

        parameters_total = pd.DataFrame(
            dtype=float, columns=["const"] + list(x.columns)
        )
        self.model_fits = pd.DataFrame(
            dtype=float, columns=["RMSE", "CV-RMSE", "deltaRMSE", "r2"]
        )

        for _ in range(x.shape[1] - 1):  # x.columns[1:]
            x_data, parameters, model = self._get_model(x, y)

            # Copy coefficients
            n_parameters = len(x_data.columns)
            parameters_total.loc[n_parameters, parameters.index] = parameters[
                "coefficients"
            ]

            # Calculate errors
            self.model_fits.loc[n_parameters, "RMSE"] = rmse(
                y, model.predict(sm.add_constant(x_data))
            )
            cross_validation = cross_validate(
                _statsmodel_wrapper(sm.OLS),  # LinearRegression(),
                x_data,
                y,
                cv=int(1 / crossvalidation_split),
                scoring=("neg_mean_squared_error"),
            )
            self.model_fits.loc[n_parameters, "CV-RMSE"] = np.sqrt(
                abs(cross_validation["test_score"])
            ).mean()

            self.model_fits.loc[n_parameters, "r2"] = model.rsquared
            x = x_data.copy()

        self.model_fits["deltaRMSE"] = abs(
            self.model_fits["CV-RMSE"] - self.model_fits["RMSE"]
        )
        self.coeff_total = parameters_total.rename(columns={"const": "intercept"})

        return pd.concat([self.coeff_total, self.model_fits], axis=1)

    def select_predictors(self, idx: int, plot=True):
        """
        Set predictors according to the results from :py:meth:`~MagmaPEC.error_propagation.FeOi_prediction.calculate_model_fits`.

        idx : int
           model index in the results from :py:meth:`~MagmaPEC.error_propagation.FeOi_prediction.calculate_model_fits`
        """

        if not hasattr(self, "model_fits"):
            raise AttributeError(
                "model_fits attribute missing, run calculate_model_fits first!"
            )

        self.predictors = (
            self.coeff_total.loc[idx].drop("intercept").dropna().index.values
        )
        self.get_OLS_coefficients()

        if not plot:
            return

        var = self.coefficients.loc[self.predictors]
        intercept = self.coefficients["intercept"]
        xvals = self.x[var.index].mul(var, axis=1).sum(axis=1)

        xstring = " + ".join([f"{val:.2f}{name}" for name, val in var.items()])

        gp.layout(colos=gp.colors.bright)
        mm = 1 / 25.4

        fig, ax = plt.subplots(figsize=(90 * mm, 85 * mm))

        ax.plot(
            xvals.sort_values(),
            xvals.sort_values() + intercept,
            "--",
            c="k",
            label="regression",
        )
        ax.plot(xvals, self.FeO, "D", label="calibration data")

        ax.set_xlabel(xstring, size=8)
        ax.set_ylabel("FeO (wt.%)")

        ax.legend(fancybox=False, frameon=True)

        plt.show()

    @staticmethod
    def _FeO_initial_func(composition, coefficients):
        if isinstance(composition, pd.DataFrame):
            oxides = composition.columns.intersection(coefficients.index)
            return coefficients["intercept"] + composition[oxides].mul(
                coefficients.loc[oxides], axis=1
            ).sum(axis=1).astype(np.float32)
        elif isinstance(composition, pd.Series):
            oxides = composition.index.intersection(coefficients.index)
            return (
                coefficients["intercept"]
                + composition[oxides].mul(coefficients.loc[oxides]).sum()
            ).astype(np.float32)

    def predict(self, melt: pd.DataFrame) -> pd.Series:
        """Predict melt FeO contents with the current model"""
        return self.model(melt)


# A wrapper for statsmodel, for use within sklearn
# Not really needed since LinearRegression produces the same results as sm.OLS
class _statsmodel_wrapper(BaseEstimator, RegressorMixin):
    """A universal sklearn-style wrapper for statsmodels regressors"""

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)
