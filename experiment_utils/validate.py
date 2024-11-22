"""
Validation functions that help catch common errors in analytic pipelines. Use these functions within transformers
to ensure the input data is valid.
"""


def validate_column_presence(df, columns):
    """
    This method will raise an exception if the provided DF does not have the required columns
    """

    missing_columns = set(columns) - set(df.columns)
    if len(missing_columns) > 0:
        raise ValueError("Missing required columns: %s" % missing_columns)

    return True


def validate_column_uniqueness(df, columns):
    """
    This method will raise an error if the given columns of the provided DF do not have have unique values
    """

    for col in columns:
        duplicate_rows = df.groupBy(col).count().filter('count > 1').count()
        if duplicate_rows > 0:
            raise ValueError("The following column is required to be unique and is not: %s" % col)

    return True