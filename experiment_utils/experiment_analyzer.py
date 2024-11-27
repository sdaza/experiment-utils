"""
Class ExperimentAnlyzer to analyze and design experiments
"""

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from spark_instance import *
import pandas as pd
import numpy as np
from typing import Dict, List
from functools import reduce
from dowhy import CausalModel
from linearmodels.iv import IV2SLS
import statsmodels.formula.api as smf
from scipy import stats
import logging

# advance methods for meta-analysis
# from statsmodels.stats.meta_analysis import combine_effects

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    def __init__(
        self,
        data: DataFrame,
        outcomes: List,
        treatment_col: str,
        experiment_identifier: List = ["campaign_key"],
        group_col: str = None,
        covariates: List = None,
        target_ipw_effect: str = "ATT",
        adjustment: str = None,
        instrument_col: str = None,
        formula: str = None,
        pvalue_threshold: float = 0.05
    ):

        """
        Initialize an ExperimentAnalyzer object

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
        experiment_identifier : List
            List of columns to identify an experiment, by default ["campaign_key"]
        target_ipw_effect : str, optional
            Target IPW effect (ATT, ATE, ATC), by default "ATT"
        adjustment : str, optional
            Adjustment method, by default None
        instrument_col : str, optional
            Column name for the instrument variable, by default None
        formula : str, optional
            Formula for the estimated regression, by default None
        pvalue_threshold : float, optional
            P-value threshold for statistical significance, by default 0.05
        """
        
        self.data = data
        self.outcomes = outcomes
        self.covariates = covariates
        self.treatment_col = treatment_col
        self.group_col = group_col
        self.experiment_identifier = experiment_identifier
        self.target_ipw_effect = target_ipw_effect
        self.adjustment = adjustment
        self.instrument_col = instrument_col
        self.__check_input()
        self.formula = None
        self.pvalue_threshold = pvalue_threshold

        self.target_weights = {"ATT": "tips_stabilized_weight", 
                               "ATE": "ipw_stabilized_weight", 
                               "ATC" : "cips_stabilized_weight"}


    def __check_input(self):
        # Ensure all required columns are present in the dataframe

        if self.group_col is None:
            self.data = self.data.withColumn('group', F.lit('all'))
            self.group_col = 'group'

        required_columns = (
            self.experiment_identifier + 
            [self.treatment_col, self.group_col] + 
            self.outcomes + 
            (self.covariates if self.covariates is not None else []) + 
            ([self.instrument_col] if self.instrument_col is not None else [])
        )
        
        missing_columns = set(required_columns) - set(self.data.columns)
        
        if missing_columns:
            self.__log_and_raise_error(
                f"The following required columns are missing from the dataframe: {missing_columns}"
            )
        if self.covariates==None:
            logger.warning("No covariates specified, balance can't be assessed!")

        self.data = self.data.select(*required_columns)


    def __get_binary_covariates(self, data):

        binary_covariates = []
        if self.covariates is not None:
            for c in self.covariates:
                if data[c].nunique() == 2 and data[c].max() == 1:
                    binary_covariates.append(c)
        return binary_covariates

    def __get_numeric_covariates(self, data):
        numeric_covariates = []
        if self.covariates is not None: 
            for c in self.covariates:
                if data[c].nunique() > 2:
                    numeric_covariates.append(c)
        return numeric_covariates


    def impute_missing_values(self, data, num_covariates=None, bin_covariates=None):

        for cov in num_covariates:
            if data[cov].isna().all():
                self.__log_and_raise_error(f'Column {cov} has only missing values')
            data[cov] = data[cov].fillna(data[cov].mean())

        for cov in bin_covariates:
            if data[cov].isna().all():
                self.__log_and_raise_error(f'Column {cov} has only missing values.')
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


    def linear_regression(self, data: pd.DataFrame, outcome_variable: str, formula: str = None) -> Dict:
        """
        Runs a linear regression of the outcome variable on the treatment variable.

        Parameters
        ----------
        data : pd.DataFrame
            Data to run the regression on
        outcome_variable : str
            Name of the outcome variable
        formula : str, optional
            Regression formula, by default None

        Returns
        -------
        Dict
            Regression results
        """
        

        formula = f"{outcome_variable} ~ {self.treatment_col}"
        self.formula = formula
        model = smf.ols(formula, data=data)
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self.treatment_col]
        intercept = results.params["Intercept"]
        relative_uplift = coefficient / intercept
        standard_error = results.bse[self.treatment_col]
        pvalue = results.pvalues[self.treatment_col]

        return {
            "group": data[self.group_col].unique()[0],
            "outcome": outcome_variable,
            "treatment_members": data[self.treatment_col].sum(),
            "control_members": data[self.treatment_col].count() - data[self.treatment_col].sum(),
            "control_value": intercept,
            "treatment_value": intercept+coefficient,
            "absolute_uplift": coefficient,
            "relative_uplift": relative_uplift,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self.pvalue_threshold else 0
        }


    def weighted_least_squares(self, data: pd.DataFrame, outcome_variable: str, formula: str = None) -> Dict:
        """
        Runs a weighted least squares regression of the outcome variable on the treatment variable.

        Parameters
        ----------
        data : pd.DataFrame
            Data to run the regression on
        outcome_variable : str
            Name of the outcome variable
        formula : str, optional
            Regression formula, by default None
        Returns
        -------
        Dict
            Regression results
        """

        if formula is None:
            formula = f"{outcome_variable} ~ 1 + {self.treatment_col}"
        else: 
            formula = self.formula

        model = smf.wls(
            formula,
            data=data,
            weights=data[self.target_weights[self.target_ipw_effect]],
        )
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self.treatment_col]
        intercept = results.params["Intercept"]
        relative_uplift = coefficient / intercept
        standard_error = results.bse[self.treatment_col]
        pvalue = results.pvalues[self.treatment_col]

        return {
            "group": data[self.group_col].unique()[0],
            "outcome": outcome_variable,
            "treatment_members": data[self.treatment_col].sum(),
            "control_members": data[self.treatment_col].count() - data[self.treatment_col].sum(),
            "control_value": intercept,
            "treatment_value": intercept+coefficient,
            "absolute_uplift": coefficient,
            "relative_uplift": relative_uplift,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self.pvalue_threshold else 0
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


    def iv_regression(self, data: pd.DataFrame, outcome_variable: str, formula: str = None) -> Dict:
        """
        Perform instrumental variable regression of the outcome variable on the treatment variable.

        Parameters
        ----------
        data : pd.DataFrame
            Data to run the regression on
        outcome_variable : str
            Name of the outcome variable
        formula : str, optional
            Regression formula, by default None

        Returns
        -------
        Dict
            Regression results including absolute and relative uplift, standard error, and p-value
        """

        if not self.instrument_col:
            self.__log_and_raise_error("Instrument column must be specified for IV adjustment")

        if formula is None:
            formula = f"{outcome_variable} ~ 1 + [{self.treatment_col} ~ {self.instrument_col}]"
        else: 
            formula = self.formula

        model = IV2SLS.from_formula(formula, data)
        results = model.fit(cov_type='robust')

        coefficient = results.params[self.treatment_col]
        intercept = results.params["Intercept"]
        relative_uplift = coefficient / intercept
        standard_error = results.std_errors[self.treatment_col]
        pvalue = results.pvalues[self.treatment_col]

        return {
            "group": data[self.group_col].unique()[0],
            "outcome": outcome_variable,
            "treatment_members": data[self.treatment_col].sum(),
            "control_members": data[self.treatment_col].count() - data[self.treatment_col].sum(),
            "control_value": intercept,
            "treatment_value": intercept+coefficient,
            "absolute_uplift": coefficient,
            "relative_uplift": relative_uplift,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self.pvalue_threshold else 0
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
                    "mean_treated": mean_treated,
                    "mean_control": mean_control,
                    "smd": smd,
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

        key_experiments = self.data.select(*self.experiment_identifier).distinct().collect()
        results = []
        
        if adjustment is None: 
            adjustment = self.adjustment

        # iterate over each combination of experimental units
        for row in key_experiments:

            logger.warning(f'Processing: {row}')
            filter_condition = reduce(
                lambda a, b: a & b,
                [
                    (F.col(unit) == row[unit])
                    for unit in self.experiment_identifier
                ],
            )

            temp = self.data.filter(filter_condition)
            temp_pd = temp.toPandas()
            numeric_covariates = self.__get_numeric_covariates(data=temp_pd)
            binary_covariates = self.__get_binary_covariates(data=temp_pd)
            groups = temp_pd[self.group_col].unique()

            for group in groups:
                temp_group = temp_pd[temp_pd[self.group_col] == group].copy()

                groupvalues = set(temp_group[self.treatment_col].unique())
                if len(groupvalues) != 2:
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

                
                if len(final_covariates)==0 & len(self.covariates if self.covariates is not None else [])>0:
                    logger.warning("No valid covariates, balance can't be assessed!")

                if len(final_covariates) > 0:
                    temp_group["weights"] = 1
                    temp_group = self.standardize_covariates(
                        temp_group, final_covariates
                    )
                    
                    balance = self.calculate_smd(
                        data=temp_group, covariates=final_covariates
                    )

                    logger.warning(
                        f'::::: Balance {group}: {np.round(balance["balance_flag"].mean(), 2)}'
                    )

                    imbalance = balance[balance.balance_flag==0]
                    if imbalance.shape[0] > 0:
                        logger.warning('::::: Imbalanced covariates')
                        logger.warning(imbalance[['covariate', 'smd', 'balance_flag']])

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

                        logger.warning(
                            f'::::: Adjusted balance {group}: {np.round(adjusted_balance["balance_flag"].mean(), 2)}'
                        )
                        adj_imbalance = adjusted_balance[adjusted_balance.balance_flag==0]
                        if adj_imbalance.shape[0] > 0:
                            logger.warning('::::: Imbalanced covariates')
                            logger.warning(adj_imbalance[['covariate', 'smd', 'balance_flag']])
    

                models = {
                    None: self.linear_regression,
                    "IPW": self.weighted_least_squares,
                    "IV": self.iv_regression,
                }

                for outcome in self.outcomes:
                    output = models[adjustment](data=temp_group, outcome_variable=outcome)
                    output['adjustment'] = 'No adjustment' if adjustment is None else adjustment
                    if adjustment == 'IPW':
                        output['balance'] = np.round(adjusted_balance['balance_flag'].mean(), 2)
                    elif len(final_covariates)>0:
                        output['balance'] = np.round(balance['balance_flag'].mean(), 2)
                    output['experiment'] = list(row.asDict().values())
                    results.append(output)

        result_columns = ['experiment', 'group', 'outcome',  'adjustment',
                          'treatment_members', 'control_members', 'control_value', 
                          'treatment_value', 'absolute_uplift', 'relative_uplift', 
                          'stat_significance', 'standard_error', 
                          'pvalue']

        if len(final_covariates) > 0:
            index_to_insert = result_columns.index('adjustment') + 1
            result_columns.insert(index_to_insert, 'balance')

        self.results = pd.DataFrame(results)[result_columns]


    def combine_results(self, grouping_cols=['group', 'outcome']):
        """
        Combine results across experiments using fixed effects meta-analysis.

        Parameters
        ----------
        grouping_cols : list, optional
            The columns to group by. Defaults to ['group', 'outcome']
        effect : str, optional
            The method to use for combining results (fixed or random). Defaults to 'fixed'.    

        Returns
        -------
        A Pandas DataFrame with combined results
        """


        pooled_results = self.results.groupby(grouping_cols).apply(
            lambda df: pd.Series(self.__get_combined_estimate(df))
        ).reset_index()

        result_columns = grouping_cols + ['treatment_members', 'control_members', 
                                          'combined_absolute_uplift', 'combined_relative_uplift', 
                                          'stat_significance', 'standard_error', 'pvalue']
        if 'balance' in self.results.columns:
            index_to_insert = len(grouping_cols)
            result_columns.insert(index_to_insert, 'average_balance')
        pooled_results['stat_significance'] = pooled_results['stat_significance'].astype(int)
        return pooled_results[result_columns]


    def __get_combined_estimate(self, data):
        weights = 1 / (data['standard_error'] ** 2)
        absolute_estimate = np.sum(weights * data['absolute_uplift']) / np.sum(weights)
        pooled_standard_error = np.sqrt(1 / np.sum(weights))
        relative_estimate = np.sum(weights * data['relative_uplift']) / np.sum(weights)

        results = {
            'treatment_members': data['treatment_members'].sum(),
            'control_members': data['control_members'].sum(),
            'combined_absolute_uplift': absolute_estimate,
            'combined_relative_uplift': relative_estimate,
            'standard_error': pooled_standard_error,
            'pvalue': stats.norm.sf(abs(absolute_estimate/ pooled_standard_error)) * 2
        }
        if 'balance' in data.columns:
            results['average_balance'] = data['balance'].mean()
        results['stat_significance'] = 1 if results['pvalue'] < self.pvalue_threshold else 0
    
        return results
    

    def __log_and_raise_error(self, message, exception_type=ValueError):
        """
        Logs an error message and raises an exception of the specified type.

        :param message: The error message to log and raise.
        :param exception_type: The type of exception to raise (default is ValueError).
        """
        
        logger.error(message)
        raise exception_type(message)