"""
Collection of helper methods. These should be fully generic and make no
assumptions about the format of input data.
"""

import logging
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from typing import Iterable

def turn_off_package_logger(package: str):
    to_logger = logging.getLogger(package)
    to_logger.setLevel(logging.ERROR)
    to_logger.handlers = [logging.NullHandler()]


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # clear existing handlers
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_and_raise_error(logger, message, exception_type=ValueError):
    """"
    Logs an error message and raises an exception of the specified type.

    :param message: The error message to log and raise.
    :param exception_type: The type of exception to raise (default is ValueError).
    """

    logger.error(message)
    raise exception_type(message)


def melt_df(
        df: DataFrame,
        id_vars: Iterable[str], value_vars: Iterable[str] = None,
        var_name: str = "variable", value_name: str = "value") -> DataFrame:
    """Convert :class:`DataFrame` from wide to long format."""

    if value_vars is None:
        value_vars = set(df.columns) - set(id_vars)

    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = array(*(
        struct(F.lit(c).alias(var_name), F.col(c).alias(value_name))
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))

    cols = id_vars + [col("_vars_and_vals")[x].alias(x)
                      for x in [var_name, value_name]]
    return _tmp.select(*cols)


def rename_columns(df, list_old_names, list_new_names):
    """Rename list of columns"""
    for old, new in zip(list_old_names, list_new_names):
        df = df.withColumnRenamed(old, new)
    return df


def remove_duplicates(data, cols=[], order_cols=[]):
    """Deterministic removal of duplicate rows"""
    window_spec = Window.partitionBy(*cols).orderBy(*order_cols)
    data = data.withColumn('row_num', F.row_number().over(window_spec))
    data = data.filter(F.col('row_num') == 1).drop('row_num')
    return data


def create_dataframe_from_dict(d, column_names=None):
    """Create dataframe from dictionary (map of values)"""
    data = [(key, value) for key, values in d.items() for value in values]
    return get_spark().createDataFrame(data, column_names)


def generate_bootstrap_samples(df=None, n_samples=None, facets=None, id_vars=None):
        '''
        Generate bootstrap samples for user dataframe.
        
        Parameters
        ---------- 
        df : pd.DataFrame
            User dataframe.
        id_vars : list
            List of id columns.
        n_samples : int
            Number of samples.
        '''

        print('Generating bootstrap samples...')

        # remove duplicates
        df = df.select((facets or []) + [id_vars]).dropDuplicates([id_vars])

        # generate a strata ID number and join back onto the main dataframe.
        if facets:
            strata_id = (
                df
                .select(facets or [])
                .dropDuplicates(subset=facets or [])
                .withColumn('strata', F.monotonically_increasing_id())
            )
            df = df.join(strata_id, on=facets or [])
        else:
            df = df.withColumn('strata', F.lit(0))

        # RDD of strata id for each user (order of columns is important)
        user_strata = df.select('strata', id_vars).rdd.map(tuple)

        # list of strata
        strata_list = (
            df
            .select('strata')
            .dropDuplicates()
            .orderBy('strata')
            .rdd.map(lambda x: x['strata']).collect()
        )

        # list of fraction to be sampled for each stratum
        # all strata are given equal weight
        fractions = [1.0] * len(strata_list)

        # dictionary of strata + fraction to be sampled
        strata_fractions = dict(zip(strata_list, fractions))

        # Initialize empty dataframe
        field = [T.StructField("strata", T.IntegerType(), True),
                T.StructField(id_vars, T.IntegerType(), True),
                T.StructField("index_sample", T.IntegerType(), True)]

        schema = T.StructType(field)
        bootstrap_samples = (
            get_spark()
            .createDataFrame(get_spark().sparkContext.emptyRDD(), schema)
        )

        # sample N times
        for n in range(n_samples):
            # seed is None so that there is variation in the samples
            sample = (
                user_strata
                .sampleByKey(withReplacement=True,
                            fractions=strata_fractions,
                            seed=None)
            )
            sample = sample.toDF(['strata', id_vars])
            sample = sample.withColumn('index_sample', F.lit(n))
            bootstrap_samples = bootstrap_samples.union(sample)

        return bootstrap_samples


def map_column(df, column, mapping, new_column):
    """Map column to a new column"""
    df = df.withColumn(new_column, F.lit(None))
    for key, value in mapping.items():
        df = df.withColumn(new_column, F.when(F.col(column) == key, F.lit(value)).otherwise(F.col(new_column)))
    return df