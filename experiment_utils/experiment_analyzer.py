from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from typing import Dict, List
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

        self.target_weights = {"ATT": "tips_stabilized_weight"}

    def __check_input(self):
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
            if data.select(c).distinct().count() == 2 and data.agg({c: 'max'}).collect()[0][0] == 1:
                binary_covariates.append(c)
        return binary_covariates

    def __get_numeric_covariates(self, data):
        numeric_covariates = []
        for c in self.covariates:
            if data.select(c).distinct().count() > 2:
                numeric_covariates.append(c)
        return numeric_covariates

    def impute_missing_values(self, data, num_covariates=None, bin_covariates=None):
        for cov in num_covariates:
            mean_value = data.agg({cov: 'mean'}).collect()[0][0]
            data = data.fillna({cov: mean_value})

        for cov in bin_covariates:
            mode_value = data.groupBy(cov).count().orderBy(F.desc('count')).first()[0]
            data = data.fillna({cov: mode_value})

        return data

    def standardize_covariates(self, data: DataFrame, covariates: List[str]) -> DataFrame:
        for covariate in covariates:
            mean = data.agg(F.mean(covariate)).collect()[0][0]
            stddev = data.agg(F.stddev(covariate)).collect()[0][0]
            data = data.withColumn(f"z_{covariate}", (F.col(covariate) - mean) / stddev)
        return data

    def calculate_smd(self, data=None, covariates=None, weights_col="weights", threshold=0.1):
        treated = data.filter(F.col(self.treatment_col) == 1)
        control = data.filter(F.col(self.treatment_col) == 0)

        if covariates is None:
            covariates = self.numeric_covariates + self.binary_covariates

        smd_results = []
        for cov in covariates:
            mean_treated = treated.agg(F.avg(cov)).collect()[0][0]
            mean_control = control.agg(F.avg(cov)).collect()[0][0]

            var_treated = treated.agg(F.variance(cov)).collect()[0][0]
            var_control = control.agg(F.variance(cov)).collect()[0][0]

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
        key_experiments = self.data.select(self.experimental_units).distinct().collect()
        results = []
        
        if adjustment is None: 
            adjustment = self.adjustment

        for row in key_experiments:
            filter_condition = reduce(
                lambda a, b: a & b,
                [
                    (F.col(unit) == row[unit])
                    for unit in self.experimental_units
                ],
            )

            temp = self.data.filter(filter_condition)
            temp_pd = temp.toPandas()
            numeric_covariates = self.__get_numeric_covariates(data=temp)
            binary_covariates = self.__get_binary_covariates(data=temp)
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

                balance = self.calculate_smd(
                    data=temp_group, covariates=final_covariates
                )

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