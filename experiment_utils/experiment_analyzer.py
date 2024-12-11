"""
ExperimentAnalyzer class to analyze and design experiments
"""

import logging
from typing import Dict, List
from functools import reduce
import pandas as pd
import numpy as np
from dowhy import CausalModel
from linearmodels.iv import IV2SLS
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import gaussian_kde
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from .utils import turn_off_package_logger, log_and_raise_error
from .spark_instance import *


class ExperimentAnalyzer:
    """
    Class ExperimentAnlyzer to analyze and design experiments
    """

    def __init__(
        self,
        data: DataFrame,
        outcomes: List,
        treatment_col: str,
        experiment_identifier: List = None,
        covariates: List = None,
        target_ipw_effect: str = "ATT",
        adjustment: str = None,
        instrument_col: str = None,
        alpha: float = 0.05,
        regression_covariates: List = None,
        assess_overlap=False):

        """
        Initialize ExperimentAnalyzer

        Parameters
        ----------
        data : DataFrame
            PySpark Dataframe
        outcomes : List
            List of outcome variables
        covariates : List
            List of covariates
        treatment_col : str
            Column name for the treatment variable
        experiment_identifier : List
            List of columns to identify an experiment
        target_ipw_effect : str, optional
            Target IPW effect (ATT, ATE, ATC), by default "ATT"
        assess_overlap : bool, optional
            Assess overlap between treatment and control groups (slow) when using IPW to adjust covariates, by default False
        adjustment : str, optional
            Adjustment method, by default None
        instrument_col : str, optional
            Column name for the instrument variable, by default None
        alpha : float, optional
            Significance level, by default 0.05
        regression_covariates : List, optional
            List of covariates to include in the final linear regression model, by default None
        """

        self.logger = logging.getLogger('Experiment Analyzer')
        self.logger.setLevel(logging.INFO)
        self.data = self.__ensure_spark_df(data)
        self.outcomes = self.__ensure_list(outcomes)
        self.covariates = self.__ensure_list(covariates)
        self.treatment_col = treatment_col
        self.experiment_identifier = self.__ensure_list(experiment_identifier)
        self.target_ipw_effect = target_ipw_effect
        self.assess_overlap = assess_overlap
        self.adjustment = adjustment
        self.instrument_col = instrument_col
        self.regression_covariates = self.__ensure_list(regression_covariates)
        self.__check_input()
        self.alpha = alpha
        self._results = None
        self._balance = []
        self._adjusted_balance = []
        self.final_covariates = []

        self.target_weights = {"ATT": "tips_stabilized_weight",
                               "ATE": "ipw_stabilized_weight",
                               "ATC": "cips_stabilized_weight"}

    def __check_input(self):

        # dataframe is empty
        if self.data.isEmpty():
            log_and_raise_error(self.logger, "Dataframe is empty!")

        # impute covariates from regression covariates
        if (len(self.covariates) == 0) & (len(self.regression_covariates) > 0):
            self.covariates = self.regression_covariates

        # regression covariates has to be a subset of covariates
        if len(self.regression_covariates) > 0:
            if not set(self.regression_covariates).issubset(set(self.covariates)):
                log_and_raise_error(self.logger, "Regression covariates should be a subset of covariates")

        # check if all required columns are present
        required_columns = (self.experiment_identifier + [self.treatment_col] + self.outcomes +
                            self.covariates + ([self.instrument_col] if self.instrument_col is not None else []))

        missing_columns = set(required_columns) - set(self.data.columns)

        if missing_columns:
            log_and_raise_error(self.logger, f"The following required columns are missing from the dataframe: {missing_columns}")
        if len(self.covariates) == 0:
            self.logger.warning("No covariates specified, balance can't be assessed!")

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
        """"
        Impute missing values for numeric and binary covariates
        """
        for cov in num_covariates:
            if data[cov].isna().all():
                log_and_raise_error(self.logger, f'Column {cov} has only missing values')
            data[cov] = data[cov].fillna(data[cov].mean())

        for cov in bin_covariates:
            if data[cov].isna().all():
                log_and_raise_error(self.logger, f'Column {cov} has only missing values.')
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

    def __create_formula(self, outcome_variable, type: str = 'regression'):
        """
        Create formula for final regression model
        """

        formula_dict = {
            'regression': f"{outcome_variable} ~ 1 + {self.treatment_col}",
            'iv': f"{outcome_variable} ~ 1 + [{self.treatment_col} ~ {self.instrument_col}]"
        }
        reg_covs = list(set(self.final_covariates) & set(self.regression_covariates))

        if len(reg_covs) > 0:
            zreg_covs = [f"z_{cov}" for cov in reg_covs]
            formula = formula_dict[type] +  ' + ' + ' + '.join(zreg_covs)
        else:
            formula = formula_dict[type]

        return formula

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

        formula = self.__create_formula(outcome_variable=outcome_variable)
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

        formula = self.__create_formula(outcome_variable=outcome_variable)
        model = smf.wls(
            formula,
            data=data,
            weights=data[self.target_weights[self.target_ipw_effect]],
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
            log_and_raise_error(self.logger, "Instrument column must be specified for IV adjustment")

        formula = self.__create_formula(outcome_variable=outcome_variable, type='iv')
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

        turn_off_package_logger('dowhy')

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
            covariates = self.final_covariates

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

    def get_overlap_coefficient(self, treatment_scores, control_scores, grid_points=1000, bw_method=None):
        """
        Calculate the Overlap Coefficient between treatment and control propensity scores.

        Parameters
        ----------
        treatment_scores : array-like
            Array of treatment propensity scores.
        control_scores : array-like
            Array of control propensity scores.
        grid_points : int, optional
            number of points to evaluate KDE on (default is 10000 for higher resolution)
        bw_method : float, optional
            Bandwidth method for the KDE estimation. Defaults to 0.1

        Returns
        -------
        float
        """

        kde_treatment = gaussian_kde(treatment_scores, bw_method=bw_method)
        kde_control = gaussian_kde(control_scores, bw_method=bw_method)

        min_score = min(treatment_scores.min(), control_scores.min())
        max_score = max(treatment_scores.max(), control_scores.max())
        x_grid = np.linspace(min_score, max_score, grid_points)
        kde_treatment_values = kde_treatment(x_grid)
        kde_control_values = kde_control(x_grid)

        overlap_coefficient = np.trapz(np.minimum(kde_treatment_values, kde_control_values), x_grid)

        return overlap_coefficient

    def get_effects(self, min_binary_count=100, adjustment=None):
        """
        Calculate effects (uplifts), given the data and experimental units.

        Parameters
        ----------
        min_binary_count : int, optional
            The minimum number of observations required for a binary covariate to be included in the analysis. Defaults to 100.
        adjustment : str, optional
            The type of adjustment to apply to estimation: 'IPW', 'IV'. Default is None.

        Returns
        -------
        results: A Pandas DataFrame with effects.
        balance: A Pandas DataFrame with balance metrics.
        adjusted_balance: A Pandas DataFrame with adjusted balance metrics.
        imbalance: A Pandas DataFrame with imbalance covariates.
        """

        models = {
            None: self.linear_regression,
            "IPW": self.weighted_least_squares,
            "IV": self.iv_regression,
        }

        key_experiments = self.data.select(*self.experiment_identifier).distinct().collect()

        temp_results = []

        if adjustment is None:
            adjustment = self.adjustment

        self._balance = []
        self._adjusted_balance = []

        # iterate over each combination of experimental units
        for row in key_experiments:

            experiment_tuple = tuple(row.asDict().values())
            self.logger.info('Processing: %s', row)
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

            treatvalues = set(temp_pd[self.treatment_col].unique())
            if len(treatvalues) != 2:
                self.logger.warning('Skipping as there are no valid treatment-control groups!')
                continue
            if not (0 in treatvalues and 1 in treatvalues):
                log_and_raise_error(self.logger, f'The treatment column {self.treatment_col} must be 0 and 1')

            temp_pd = self.impute_missing_values(
                data=temp_pd,
                num_covariates=numeric_covariates,
                bin_covariates=binary_covariates,
            )

            # remove constant or low frequency covariates
            numeric_covariates = [
                c for c in numeric_covariates if temp_pd[c].std() != 0
            ]
            binary_covariates = [
                c
                for c in binary_covariates
                if temp_pd[c].sum() >= min_binary_count
            ]
            binary_covariates = [
                c for c in binary_covariates if temp_pd[c].std() != 0
            ]

            final_covariates = numeric_covariates + binary_covariates
            self.final_covariates = final_covariates
            if len(final_covariates) == 0 & len(self.covariates if self.covariates is not None else []) > 0:
                self.logger.warning("No valid covariates, balance can't be assessed!")

            if len(final_covariates) > 0:
                temp_pd["weights"] = 1
                temp_pd = self.standardize_covariates(temp_pd, final_covariates)
                balance = self.calculate_smd(
                    data=temp_pd, covariates=final_covariates
                )
                balance["experiment"] = [experiment_tuple] * len(balance)
                balance = self.__transform_tuple_column(balance, "experiment", self.experiment_identifier)
                self._balance.append(balance)
                self.logger.info('::::: Balance: %.2f', np.round(balance["balance_flag"].mean(), 2))
                if adjustment == "IPW":
                    temp_pd = self.estimate_ipw(
                        data=temp_pd,
                        covariates=[f"z_{cov}" for cov in final_covariates],
                        outcome_variable=self.outcomes[0],
                    )

                    adjusted_balance = self.calculate_smd(
                        data=temp_pd,
                        covariates=final_covariates,
                        weights_col=self.target_weights[self.target_ipw_effect]
                    )
                    adjusted_balance["experiment"] = [experiment_tuple] * len(adjusted_balance)
                    adjusted_balance = self.__transform_tuple_column(
                        adjusted_balance, "experiment", self.experiment_identifier)
                    self._adjusted_balance.append(adjusted_balance)

                    self.logger.info(
                        '::::: Adjusted balance: %.2f',
                        np.round(adjusted_balance["balance_flag"].mean(), 2)
                    )
                    if self.assess_overlap:
                        overlap = self.get_overlap_coefficient(
                            temp_pd[temp_pd[self.treatment_col] == 1].propensity_score,
                            temp_pd[temp_pd[self.treatment_col] == 0].propensity_score)
                        self.logger.info('::::: Overlap: %.2f', np.round(overlap, 2))

            for outcome in self.outcomes:
                output = models[adjustment](data=temp_pd, outcome_variable=outcome)
                output['adjustment'] = 'No adjustment' if adjustment is None else adjustment
                if adjustment == 'IPW':
                    output['balance'] = np.round(adjusted_balance['balance_flag'].mean(), 2)
                elif (len(final_covariates) > 0):
                    output['balance'] = np.round(balance['balance_flag'].mean(), 2)
                output['experiment'] = experiment_tuple

                temp_results.append(output)

        result_columns = ['experiment', 'outcome', 'adjustment',
                          'treated_units', 'control_units', 'control_value',
                          'treatment_value', 'absolute_effect', 'relative_effect',
                          'stat_significance', 'standard_error',
                          'pvalue']

        if len(final_covariates) > 0:
            index_to_insert = result_columns.index('adjustment') + 1
            result_columns.insert(index_to_insert, 'balance')

        clean_temp_results = pd.DataFrame(temp_results)
        clean_temp_results = clean_temp_results[result_columns]

        if len(self._balance) > 0:
            self._balance = pd.concat(self._balance)
        if len(self._adjusted_balance) > 0:
            self._adjusted_balance = pd.concat(self._adjusted_balance)

        self._results = self.__transform_tuple_column(clean_temp_results, 'experiment', self.experiment_identifier)

    def combine_effects(self, data: pd.DataFrame = None, grouping_cols: List = None):
        """
        Combine effects across experiments using fixed effects meta-analysis.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The DataFrame containing the results. Defaults to self._results
        grouping_cols : list, optional
            The columns to group by. Defaults to experiment_identifer + ['outcome']
        effect : str, optional
            The method to use for combining results (fixed or random). Defaults to 'fixed'.

        Returns
        -------
        A Pandas DataFrame with combined results
        """

        if data is None:
            data = self._results

        if grouping_cols is None:
            self.logger.warning('No grouping columns specified, using only outcome!')
            grouping_cols = ['outcome']
        else:
            grouping_cols = self.__ensure_list(grouping_cols)
            if 'outcome' not in grouping_cols:
                grouping_cols.append('outcome')

        if any(data.groupby(grouping_cols).size() < 2):
            self.logger.warning('There are some combinations with only one experiment!')

        pooled_results = data.groupby(grouping_cols).apply(
            lambda df: pd.Series(self.__get_fixed_meta_analysis_estimate(df))
        ).reset_index()

        result_columns = grouping_cols + ['experiments', 'treated_units', 'control_units',
                                          'absolute_effect', 'relative_effect', 'stat_significance',
                                          'standard_error', 'pvalue']
        if 'balance' in data.columns:
            index_to_insert = len(grouping_cols)
            result_columns.insert(index_to_insert + 1, 'balance')
        pooled_results['stat_significance'] = pooled_results['stat_significance'].astype(int)

        self.logger.info('Combining effects using fixed-effects meta-analysis!')
        return pooled_results[result_columns]

    def __get_fixed_meta_analysis_estimate(self, data):
        weights = 1 / (data['standard_error'] ** 2)
        absolute_estimate = np.sum(weights * data['absolute_effect']) / np.sum(weights)
        pooled_standard_error = np.sqrt(1 / np.sum(weights))
        relative_estimate = np.sum(weights * data['relative_effect']) / np.sum(weights)

        # pvalue
        np.seterr(invalid='ignore')
        try:
            pvalue = stats.norm.sf(abs(absolute_estimate / pooled_standard_error)) * 2
        except FloatingPointError:
            pvalue = np.nan

        meta_results = {
            'experiments': int(data.shape[0]),
            'treated_units': int(data['treated_units'].sum()),
            'control_units': int(data['control_units'].sum()),
            'absolute_effect': absolute_estimate,
            'relative_effect': relative_estimate,
            'standard_error': pooled_standard_error,
            'pvalue': pvalue
        }

        if 'balance' in data.columns:
            meta_results['balance'] = data['balance'].mean()
        meta_results['stat_significance'] = 1 if meta_results['pvalue'] < self.alpha else 0
        return meta_results

    def aggregate_effects(self, data: pd.DataFrame = None, grouping_cols: List = None):
        """
        Aggregate effects using a weighted average based on the size of the treatment group.

        Parameters
        ----------
        data : pd.DataFrame
        The DataFrame containing the results.
        grouping_cols : list, optional
        The columns to group by. Defaults to ['outcome']

        Returns
        -------
        A Pandas DataFrame with combined results
        """

        if data is None:
            data = self._results

        if grouping_cols is None:
            self.logger.warning('No grouping columns specified, using only outcome!')
            grouping_cols = ['outcome']
        else:
            grouping_cols = self.__ensure_list(grouping_cols)
            if 'outcome' not in grouping_cols:
                grouping_cols.append('outcome')

        aggregate_results = data.groupby(grouping_cols).apply(self.__compute_weighted_effect).reset_index()

        self.logger.info('Aggregating effects using weighted averages!')
        self.logger.info('For a better standard error estimation, use meta-analysis or `combine_effects`')

        # keep initial order
        result_columns = grouping_cols + ['experiments', 'balance']
        existing_columns = [col for col in result_columns if col in aggregate_results.columns]
        remaining_columns = [col for col in aggregate_results.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns
        return aggregate_results[final_columns]

    def __compute_weighted_effect(self, group):

        group['gweight'] = group['treated_units'].astype(int)
        absolute_effect = np.sum(group['absolute_effect'] * group['gweight']) / np.sum(group['gweight'])
        relative_effect = np.sum(group['relative_effect'] * group['gweight']) / np.sum(group['gweight'])
        variance = (group['standard_error'] ** 2) * group['gweight']

        pooled_variance = np.sum(variance) / np.sum(group['gweight'])
        combined_se = np.sqrt(pooled_variance)
        z_score = absolute_effect / combined_se
        combined_p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        output = pd.Series({
            'experiments': int(group.shape[0]),
            'treated_units': int(np.sum(group['gweight'])),
            'absolute_effect': absolute_effect,
            'relative_effect': relative_effect,
            'stat_significance': 1 if combined_p_value < self.alpha else 0,
            'standard_error': combined_se,
            'pvalue': combined_p_value
        })

        if 'balance' in group.columns:
            combined_balance = np.sum(group['balance'] * group['gweight']) / np.sum(group['gweight'])
            output['balance'] = combined_balance

        return output

    @property
    def imbalance(self):
        """
        Returns the imbalance DataFrame.
        """
        if len(self._adjusted_balance) > 0:
            ab = self.adjusted_balance[self.adjusted_balance.balance_flag == 0]
            if ab.shape[0] > 0:
                self.logger.info('Imbalance after adjustments!')
                return ab
        elif len(self._balance) > 0:
            b = self.balance[self.balance.balance_flag == 0]
            if b.shape[0] > 0:
                self.logger.info('Imbalance without adjustments!')
                return b
        else:
            pass

    def __transform_tuple_column(self, df, tuple_column, new_columns):
        """
        Transforms a column of tuples into separate columns.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the tuple column.
        tuple_column (str): The name of the column with tuples.
        new_columns (list): A list of new column names for the tuple elements.

        Returns:
        pd.DataFrame: A new DataFrame with the tuple elements as separate columns.
        """
        # Split the tuple column into new columns
        columns = [col for col in df.columns if col != tuple_column]
        df[new_columns] = pd.DataFrame(df[tuple_column].tolist(), index=df.index)

        # Reorder columns to have the new columns at the start
        ordered_columns = new_columns + columns
        df = df[ordered_columns]

        return df

    def __ensure_list(self, item):
        """Ensure the input is a list."""
        if item is None:
            return []
        return item if isinstance(item, list) else [item]

    @property
    def results(self):
        """"
        Returns the results DataFrame
        """
        if self._results is not None:
            return self._results
        self.logger.warning('Run the `get_effects` function first!')
        return None

    @property
    def balance(self):
        """"
        Returns the balance DataFrame
        """
        if len(self._balance) > 0:
            return self._balance
        self.logger.warning('No balance information available!')
        return None

    @property
    def adjusted_balance(self):
        """"
        Returns the adjusted balance DataFrame
        """
        if len(self._adjusted_balance) > 0:
            return self._adjusted_balance
        self.logger.warning('No adjusted balance information available!')
        return None

    def __ensure_spark_df(self, dataframe):
        """
        Convert a Pandas DataFrame to a PySpark DataFrame if it is a Pandas DataFrame.
        """
        if isinstance(dataframe, pd.DataFrame):
            spark_df = spark.createDataFrame(dataframe)
            return spark_df
        return dataframe
