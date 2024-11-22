
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import pandas as pd
import numpy as np
from typing import Dict, List
from functools import reduce
from dowhy import CausalModel
from linearmodels.iv import IV2SLS
import statsmodels.formula.api as smf
from scipy import stats


class ExperimentAnalyzer:
    """
    Class to analyze experiments on campaigns
    """

    def __init__(
        self,
        data: DataFrame,
        outcomes: List,
        covariates: List,
        treatment_col: str,
        group_col: str,
        experimental_units: List = ["campaign_key"],
        target_ipw_effect: str = "ATT",
        adjustment: str = None,
        instrument_col: str = None
    ):

        """
        Initialize an ExperimentAnalyzer object.

        Parameters
        ----------
        data : DataFrame
            Dataframe containing the data
        outcomes : List
            List of outcome variables
        covariates : List
            List of covariates
        treatment_col : str
            Column name for the treatment variable
        group_col : str
            Column name for the group identifier
        experimental_units : List
            List of experimental units, by default ["campaign_key"]
        target_ipw_effect : str, optional
            Target IPW effect, by default "ATT"
        adjustment : str, optional
            Adjustment method, by default None
        instrument_col : str, optional
            Column name for the instrument variable, by default None
        """
        
        self.data = data
        self.outcomes = outcomes
        self.covariates = covariates
        self.treatment_col = treatment_col
        self.group_col = group_col
        self.experimental_units = experimental_units
        self.target_ipw_effect = target_ipw_effect
        self.adjustment = adjustment
        self.instrument_col = instrument_col
        self.__check_input()

        self.target_weights = {"ATT": "tips_stabilized_weight", 
                               "ATE": "ipw_stabilized_weight", 
                               "A" : "cips_stabilized_weight"}


    def __check_input(self):
        # Ensure all required columns are present in the dataframe

        instrument_col = [self.instrument_col] if self.instrument_col is not None else []
        required_columns = (
            self.experimental_units + 
            [self.treatment_col, self.group_col] + 
            self.outcomes + 
            self.covariates + 
            instrument_col
        )

        missing_columns = set(required_columns) - set(self.data.columns)
        
        if missing_columns:
            raise ValueError(
                f"The following required columns are missing from the dataframe: {missing_columns}"
            )
        self.data = self.data.select(*required_columns)


    def __get_binary_covariates(self, data):
        binary_covariates = []
        for c in self.covariates:
            if data[c].nunique() == 2 and data[c].max() == 1:
                binary_covariates.append(c)
        return binary_covariates


    def __get_numeric_covariates(self, data):
        numeric_covariates = []
        for c in self.covariates:
            if data[c].nunique() > 2:
                numeric_covariates.append(c)
        return numeric_covariates


    def impute_missing_values(self, data, num_covariates=None, bin_covariates=None):
        for cov in num_covariates:
            data[cov] = data[cov].fillna(data[cov].mean())

        for cov in bin_covariates:
            data[cov] = data[cov].fillna(data[cov].mode()[0])

        return data


    def standardize_covariates(self, data: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
        """
        Standardize covariates in the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to standardize
        covariates : List[str]
            List of covariates to standardize

        Returns
        -------
        pd.DataFrame
            Data with standardized covariates
        """
        for covariate in covariates:
            data[f"z_{covariate}"] = (data[covariate] - data[covariate].mean()) / data[covariate].std()
        return data


    def linear_regression(self, data: pd.DataFrame, outcome_variable: str) -> Dict:
        """
        Runs a linear regression of the outcome variable on the treatment variable.

        Parameters
        ----------
        data : pd.DataFrame
            Data to run the regression on
        outcome_variable : str
            Name of the outcome variable

        Returns
        -------
        Dict
            Regression results
        """
        formula = f"{outcome_variable} ~ {self.treatment_col}"
        model = smf.ols(formula, data=data)
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self.treatment_col]
        relative_uplift = coefficient / results.params["Intercept"]
        standard_error = results.bse[self.treatment_col]
        p_value = results.pvalues[self.treatment_col]

        return {
            "group": data[self.group_col].unique()[0],
            "outcome": outcome_variable,
            "treatment_members": data[self.treatment_col].sum(),
            "absolute_uplift": coefficient,
            "relative_uplift": relative_uplift,
            "standard_error": standard_error,
            "p_value": p_value,
        }


    def weighted_least_squares(self, data: pd.DataFrame, outcome_variable: str) -> Dict:
        """
        Runs a weighted least squares regression of the outcome variable on the treatment variable.

        Parameters
        ----------
        data : pd.DataFrame
            Data to run the regression on
        outcome_variable : str
            Name of the outcome variable

        Returns
        -------
        Dict
            Regression results
        """
        formula = f"{outcome_variable} ~ 1 + {self.treatment_col}"
        model = smf.wls(
            formula,
            data=data,
            weights=data[self.target_weights[self.target_ipw_effect]],
        )
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self.treatment_col]
        relative_uplift = coefficient / results.params["Intercept"]
        standard_error = results.bse[self.treatment_col]
        p_value = results.pvalues[self.treatment_col]

        return {
            "group": data[self.group_col].unique()[0],
            "outcome": outcome_variable,
            "treatment_members": data[self.treatment_col].sum(),
            "absolute_uplift": coefficient,
            "relative_uplift": relative_uplift,
            "standard_error": standard_error,
            "p_value": p_value,
        }


    def estimate_ipw(self, data: pd.DataFrame, covariates: List[str], outcome_variable: str) -> pd.DataFrame:
        """
        Estimate the IPW using the dowhy library.

        Parameters
        ----------
        data : pd.DataFrame
            Data to estimate the IPW from
        covariates : List[str]
            List of covariates to include in the estimation
        outcome_variable : str
            Name of the outcome variable

        Returns
        -------
        pd.DataFrame
            Data with the estimated IPW
        """
        causal_model = CausalModel(
            data=data,
            treatment=self.treatment_col,
            outcome=outcome_variable,
            common_causes=covariates,
        )

        identified_estimand = causal_model.identify_effect(
            proceed_when_unidentifiable=True
        )

        causal_model.estimate_effect(
            identified_estimand,
            target_units="att",
            method_name="backdoor.propensity_score_weighting",
            method_params={
                "weighting_scheme": "ips_weight",
                "propensity_score_model_params": {
                    "max_iter": 5000,
                    "penalty": "l2",
                    "C": 1.0,
                },
            },
        )

        return data


    def iv_regression(self, data: pd.DataFrame, outcome_variable: str) -> Dict:
        """
        Perform instrumental variable regression of the outcome variable on the treatment variable.

        Parameters
        ----------
        data : pd.DataFrame
            Data to run the regression on
        outcome_variable : str
            Name of the outcome variable

        Returns
        -------
        Dict
            Regression results including absolute and relative uplift, standard error, and p-value
        """
        if not self.instrument_col:
            raise ValueError("Instrument column must be specified for IV adjustment")

        formula = f"{outcome_variable} ~ 1 + [{self.treatment_col} ~ {self.instrument_col}]"
        model = IV2SLS.from_formula(formula, data)
        results = model.fit(cov_type='robust')

        coefficient = results.params[self.treatment_col]
        relative_uplift = coefficient / results.params["Intercept"]
        standard_error = results.std_errors[self.treatment_col]
        p_value = results.pvalues[self.treatment_col]

        return {
            "group": data[self.group_col].unique()[0],
            "outcome": outcome_variable,
            "treatment_members": data[self.treatment_col].sum(),
            "absolute_uplift": coefficient,
            "relative_uplift": relative_uplift,
            "standard_error": standard_error,
            "p_value": p_value,
        }


    def calculate_smd(
        self, data=None, covariates=None, weights_col="weights", threshold=0.1
    ):
        """
        Calculate standardized mean differences (SMDs) between treatment and control groups.

        Parameters
        ----------
        data : DataFrame, optional
            DataFrame containing the data to calculate SMDs on. If None, uses the data from the class.
        covariates : list, optional
            List of column names to calculate SMDs for. If None, uses all numeric and binary covariates.
        weights_col : str, optional
            Name of the column to use for weighting the means. Defaults to 'weights'.
        threshold : float, optional
            Threshold to determine if a covariate is balanced. Defaults to 0.1.

        Returns
        -------
        DataFrame
            DataFrame containing the SMDs and balance flags for each covariate.
        """

        treated = data[data[self.treatment_col] == 1]
        control = data[data[self.treatment_col] == 0]

        if covariates is None:
            covariates = self.numeric_covariates + self.binary_covariates

        smd_results = []
        for cov in covariates:

            mean_treated = np.average(treated[cov], weights=treated[weights_col])
            mean_control = np.average(control[cov], weights=control[weights_col])

            var_treated = np.average(
                (treated[cov] - mean_treated) ** 2, weights=treated[weights_col]
            )
            var_control = np.average(
                (control[cov] - mean_control) ** 2, weights=control[weights_col]
            )

            std_treated = np.sqrt(var_treated)
            std_control = np.sqrt(var_control)

            pooled_std = np.sqrt((var_treated + var_control) / 2)

            smd = (mean_treated - mean_control) / pooled_std if pooled_std != 0 else 0

            balance_flag = 1 if abs(smd) <= threshold else 0

            smd_results.append(
                {
                    "covariate": cov,
                    "weighted_mean_treated": mean_treated,
                    "weighted_mean_control": mean_control,
                    "weighted_smd": smd,
                    "balance_flag": balance_flag,
                }
            )

        smd_df = pd.DataFrame(smd_results)

        return smd_df


    def get_uplifts(self, min_binary_count=100, adjustment=None):
        """
        Calculate uplifts (causal effects), given the data and experimental units.

        Parameters
        ----------
        min_binary_count : int, optional
            The minimum number of observations required for a binary covariate to be included in the analysis. Defaults to 100.
        adjustment : str, optional
            The type of adjustment to apply to estimation: 'IPW', 'IV'. Default is None.

        Returns
        -------
        A Pandas DataFrame with effects.
        """

        key_experiments = self.data.select(self.experimental_units).distinct().collect()
        results = []
        
        if adjustment is None: 
            adjustment = self.adjustment

        # iterate over each combination of experimental units
        for row in key_experiments:

            print(f'Processing: {row}')
            filter_condition = reduce(
                lambda a, b: a & b,
                [
                    (F.col(unit) == row[unit])
                    for unit in self.experimental_units
                ],
            )

            temp = self.data.filter(filter_condition)
            temp_pd = temp.toPandas()
            numeric_covariates = self.__get_numeric_covariates(data=temp_pd)
            binary_covariates = self.__get_binary_covariates(data=temp_pd)
            groups = temp_pd[self.group_col].unique()

            for group in groups:
                temp_group = temp_pd[temp_pd[self.group_col] == group].copy()

                group_values = set(temp_group[self.treatment_col].unique())
                if len(group_values) != 2:
                    continue

                temp_group = self.impute_missing_values(
                    data=temp_group,
                    num_covariates=numeric_covariates,
                    bin_covariates=binary_covariates,
                )

                # remove constant or low frequency covariates
                numeric_covariates = [
                    c for c in numeric_covariates if temp_group[c].std() != 0
                ]
                binary_covariates = [
                    c
                    for c in binary_covariates
                    if temp_group[c].sum() >= min_binary_count
                ]
                binary_covariates = [
                    c for c in binary_covariates if temp_group[c].std() != 0
                ]
                final_covariates = numeric_covariates + binary_covariates

                temp_group["weights"] = 1
                temp_group = self.standardize_covariates(
                    temp_group, final_covariates
                )
                print(final_covariates)
                
                balance = self.calculate_smd(
                    data=temp_group, covariates=final_covariates
                )
                print(
                    f'::::: Initial balance {group}: {np.round(balance["balance_flag"].mean(), 2)}'
                )
                imbalance = balance[balance.balance_flag==0]
                if imbalance.shape[0] > 0:
                    print(imbalance)

                if adjustment == "IPW":
                    temp_group = self.estimate_ipw(
                        data=temp_group,
                        covariates=[f"z_{cov}" for cov in final_covariates],
                        outcome_variable=self.outcomes[0],
                    )

                    adjusted_balance = self.calculate_smd(
                        data=temp_group,
                        covariates=final_covariates,
                        weights_col=self.target_weights[self.target_ipw_effect],
                    )

                    print(
                        f'::::: Adjusted balance {group}: {np.round(adjusted_balance["balance_flag"].mean(), 2)}'
                    )
                    adj_imbalance = adjusted_balance[adjusted_balance.balance_flag==0]
                    if adj_imbalance.shape[0] > 0:
                        print(adj_imbalance)


                models = {
                    None: self.linear_regression,
                    "IPW": self.weighted_least_squares,
                    "IV": self.iv_regression,
                }

                for outcome in self.outcomes:
                    results.append(
                        models[adjustment](
                            data=temp_group, outcome_variable=outcome
                        )
                    )

        self.unit_results = pd.DataFrame(results)


    def pool_results(self):
        pooled_results = self.unit_results.groupby(['group', 'outcome']).apply(
            lambda df: pd.Series(self.get_pooled_estimate(df))
        ).reset_index()

        return pooled_results


    def get_pooled_estimate(self, data):
        weights = 1 / (data['standard_error'] ** 2)
        absolute_estimate = np.sum(weights * data['absolute_uplift']) / np.sum(weights)
        pooled_standard_error = np.sqrt(1 / np.sum(weights))
        relative_estimate = np.sum(weights * data['relative_uplift']) / np.sum(weights)

        results = {
            'treatment_members': data['treatment_members'].sum(),
            'pooled_absolute_uplift': absolute_estimate,
            'pooled_relative_uplift': relative_estimate,
            'standard_error': pooled_standard_error,
            'p_value': stats.norm.sf(abs(absolute_estimate/ pooled_standard_error)) * 2
        }
        return results