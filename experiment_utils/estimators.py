""""This module contains classes for performing causal inference using various estimators."""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from typing import Dict, List, Optional, Union
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from .utils import log_and_raise_error, get_logger


class Estimators:
    """
    A class for performing causal inference using various estimators.
    """

    def __init__(self, treatment_col: str, instrument_col: Optional[str] = None,
                 target_ipw_effect: str = 'ATT', alpha: float = 0.05,
                 min_ps_score: float = 0.05, max_ps_score: float = 0.95,
                 interaction_logistic_ipw: bool = False) -> None:

        self.logger = get_logger('Estimators')
        self.treatment_col = treatment_col
        self.instrument_col = instrument_col
        self.target_ipw_effect = target_ipw_effect
        self.alpha = alpha
        self.max_ps_score = max_ps_score
        self.min_ps_score = min_ps_score
        self.interaction_logistic_ipw = interaction_logistic_ipw

    def __create_formula(self, outcome_variable: str, covariates: Optional[List[str]], model_type: str = 'regression') -> str:

        formula_dict = {
            'regression': f"{outcome_variable} ~ 1 + {self.treatment_col}",
            'iv': f"{outcome_variable} ~ 1 + [{self.treatment_col} ~ {self.instrument_col}]"
        }
        if covariates:
            standardized_covariates = [f"z_{covariate}" for covariate in covariates]
            formula = formula_dict[model_type] + ' + ' + ' + '.join(standardized_covariates)
        else:
            formula = formula_dict[model_type]
        return formula

    def linear_regression(self, data: pd.DataFrame, outcome_variable: str, covariates: Optional[List[str]] = None) -> Dict[str, Union[str, int, float]]:
        """
        Perform linear regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        covariates (List[str]): The list of covariates to include in the regression model.

        Returns:
        Dict: A dictionary containing the results of the regression, including:
            - "outcome" (str): The name of the outcome variable.
            - "treated_units" (int): The number of treated units in the data.
            - "control_units" (int): The number of control units in the data.
            - "control_value" (float): The intercept of the regression model.
            - "treatment_value" (float): The predicted value for the treatment group.
            - "absolute_effect" (float): The coefficient of the treatment variable.
            - "relative_effect" (float): The relative effect of the treatment.
            - "standard_error" (float): The standard error of the treatment coefficient.
            - "pvalue" (float): The p-value of the treatment coefficient.
            - "stat_significance" (int): Indicator of statistical significance (1 if p-value < alpha, else 0).
        """

        formula = self.__create_formula(outcome_variable=outcome_variable, covariates=covariates)
        model = smf.ols(formula, data=data)
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self.treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept
        standard_error = results.bse[self.treatment_col]
        pvalue = results.pvalues[self.treatment_col]

        return {
            "outcome": outcome_variable,
            "treated_units": data[self.treatment_col].sum(),
            "control_units": data[self.treatment_col].count() - data[self.treatment_col].sum(),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self.alpha else 0
        }

    def weighted_least_squares(self, data: pd.DataFrame, outcome_variable: str,
                               weight_column: str, covariates: Optional[List[str]] = None) -> Dict[str, Union[str, int, float]]:
        """
        Perform weighted least squares regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        weight_column (str): The name of the column containing the weights for the regression.
        covariates (List[str]): The list of covariates to include in the regression model.

        Returns:
        Dict: A dictionary containing the results of the regression, including:
            - "outcome" (str): The name of the outcome variable.
            - "treated_units" (int): The number of treated units in the data.
            - "control_units" (int): The number of control units in the data.
            - "control_value" (float): The intercept of the regression model.
            - "treatment_value" (float): The predicted value for the treatment group.
            - "absolute_effect" (float): The coefficient of the treatment variable.
            - "relative_effect" (float): The relative effect of the treatment.
            - "standard_error" (float): The standard error of the treatment coefficient.
            - "pvalue" (float): The p-value of the treatment coefficient.
            - "stat_significance" (int): Indicator of statistical significance (1 if p-value < alpha, else 0).
        """
        formula = self.__create_formula(outcome_variable=outcome_variable, covariates=covariates)
        model = smf.wls(
            formula,
            data=data,
            weights=data[weight_column],
        )
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self.treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept
        standard_error = results.bse[self.treatment_col]
        pvalue = results.pvalues[self.treatment_col]

        return {
            "outcome": outcome_variable,
            "treated_units": data[self.treatment_col].sum().astype(int),
            "control_units": (data[self.treatment_col].count() - data[self.treatment_col].sum()).astype(int),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self.alpha else 0
        }

    def iv_regression(self, data: pd.DataFrame, outcome_variable: str, covariates: Optional[List[str]] = None) -> Dict[str, Union[str, int, float]]:

        if not self.instrument_col:
            log_and_raise_error(self.logger, "Instrument column must be specified for IV adjustment")

        formula = self.__create_formula(outcome_variable=outcome_variable, model_type='iv', covariates=covariates)
        model = IV2SLS.from_formula(formula, data)
        results = model.fit(cov_type='robust')

        coefficient = results.params[self.treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept
        standard_error = results.std_errors[self.treatment_col]
        pvalue = results.pvalues[self.treatment_col]

        return {
            "outcome": outcome_variable,
            "treated_units": data[self.treatment_col].sum().astype(int),
            "control_units": (data[self.treatment_col].count() - data[self.treatment_col].sum()).astype(int),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self.alpha else 0
        }

    def ipw_logistic(self, data: pd.DataFrame, covariates: List[str], penalty: str = 'l2', C: float = 1.0, max_iter: int = 5000) -> pd.DataFrame:
        """
        Estimate the Inverse Probability Weights (IPW) using logistic regression with regularization.

        Parameters
        ----------
        data : pd.DataFrame
            Data to estimate the IPW from
        covariates : List[str]
            List of covariates to include in the estimation
        penalty : str, optional
            Regularization penalty to use in the logistic regression model, by default 'l2'
        C : float, optional
            Inverse of regularization strength, by default 1.0
        max_iter : int, optional


        Returns
        -------
        pd.DataFrame
            Data with the estimated IPW
        """

        logistic_model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter)

        if self.interaction_logistic_ipw:
            poly = PolynomialFeatures(interaction_only=True, include_bias=False)
            X = poly.fit_transform(data[covariates])
            feature_names = poly.get_feature_names_out(covariates)
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = data[covariates]

        y = data[self.treatment_col]
        logistic_model.fit(X, y)

        if not logistic_model.n_iter_[0] < logistic_model.max_iter:
            self.logger.warning("Logistic regression model did not converge. Consider increasing the number of iterations or adjusting other parameters.")

        data['propensity_score'] = logistic_model.predict_proba(X)[:, 1]
        data['propensity_score'] = np.minimum(self.max_ps_score, data['propensity_score'])
        data['propensity_score'] = np.maximum(self.min_ps_score, data['propensity_score'])

        data = self.__calculate_stabilized_weights(data)
        return data

    def ipw_xgboost(self, data: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:

        X = data[covariates]
        y = data[self.treatment_col]

        xgb_model = XGBClassifier(eval_metric='logloss')
        xgb_model.fit(X, y)

        data['propensity_score'] = xgb_model.predict_proba(X)[:, 1]
        data['propensity_score'] = np.minimum(self.max_ps_score, data['propensity_score'])
        data['propensity_score'] = np.maximum(self.min_ps_score, data['propensity_score'])
        data = self.___calculate_stabilized_weights(data)
        return data

    def __calculate_stabilized_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the stabilized weights for IPW.

        Parameters
        ----------
        data : pd.DataFrame
            Data with the estimated propensity scores

        Returns
        -------
        pd.DataFrame
            Data with the calculated stabilized weights
        """
        num_units = data.shape[0]
        p_treatment = sum(data[self.treatment_col]) / num_units

        data["ips_stabilized_weight"] = data[self.treatment_col] / data[
            "propensity_score"
        ] * p_treatment + (1 - data[self.treatment_col]) / (
            1 - data["propensity_score"]
        ) * (
            1 - p_treatment
        )
        data["tips_stabilized_weight"] = data[self.treatment_col] * p_treatment + (
            1 - data[self.treatment_col]
        ) * data["propensity_score"] / (1 - data["propensity_score"]) * (1 - p_treatment)

        data["cips_stabilized_weight"] = data[self.treatment_col] * (
            1 - data["propensity_score"]
        ) / data["propensity_score"] * p_treatment + (
            1 - data[self.treatment_col]
        ) * (
            1 - p_treatment
        )

        return data
