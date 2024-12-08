import pytest
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Spark Test") \
    .master("local[*]") \
    .getOrCreate()

# Create a simple DataFrame
data = [("Alice", 1), ("Bob", 2), ("Cathy", 3)]
columns = ["Name", "Id"]
df = spark.createDataFrame(data, columns)

# Show the DataFrame
df.show()

@pytest.fixture
def sample_data(spark):
    """Fixture for creating a sample Spark DataFrame."""
    schema = StructType([
        StructField("experiment_id", StringType(), True),
        StructField("treatment", IntegerType(), True),
        StructField("outcome1", IntegerType(), True),
        StructField("covariate1", IntegerType(), True),
    ])
    data = [
        ("exp1", 1, 10, 5),
        ("exp1", 0, 12, 6),
        ("exp2", 1, 15, 7),
        ("exp2", 0, 11, 8),
    ]
    return spark.createDataFrame(data, schema)

def test_check_input(sample_data):
    """Test the __check_input method of ExperimentAnalyzer."""
    outcomes = ["outcome1"]
    treatment_col = "treatment"
    experiment_identifier = ["experiment_id"]
    covariates = ["covariate1"]

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes=outcomes,
        treatment_col=treatment_col,
        experiment_identifier=experiment_identifier,
        covariates=covariates
    )

    # This should not raise an error since all columns are present
    try:
        analyzer._ExperimentAnalyzer__check_input()
        assert True
    except Exception as e:
        pytest.fail(f"__check_input raised an exception: {e}")

def test_missing_columns(sample_data):
    """Test the __check_input method with missing columns."""
    outcomes = ["outcome1"]
    treatment_col = "treatment"
    experiment_identifier = ["experiment_id"]
    covariates = ["missing_covariate"]

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes=outcomes,
        treatment_col=treatment_col,
        experiment_identifier=experiment_identifier,
        covariates=covariates
    )

    # Expecting an error due to missing covariate
    with pytest.raises(Exception, match="The following required columns are missing from the dataframe"):
        analyzer._ExperimentAnalyzer__check_input()
