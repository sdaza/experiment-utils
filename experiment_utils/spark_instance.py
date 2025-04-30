"""
Spark instance
"""

from typing import Optional

from pyspark.sql import SparkSession


class SparkInstance:
    """Singleton class to create a Spark session"""
    _instance: Optional['SparkInstance'] = None
    spark: SparkSession | None = None  # Initialize as None

    def __new__(cls) -> 'SparkInstance':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Do not create SparkSession here anymore
        return cls._instance

    def get_spark(self) -> SparkSession:
        """Get the Spark session, creating it if it doesn't exist"""
        if self.spark is None:  # Check if already created
            # Create SparkSession here
            self.spark = (
                SparkSession.builder.master('local[*]')  # Use local[*]
                .appName('experiment_utils')
                # Consider adding specific configs if needed, e.g., for memory
                .getOrCreate()
            )
        return self.spark


# Remove module-level instantiation
spark_instance = SparkInstance()
spark = spark_instance.get_spark()

# Add a helper function for easy access
def get_spark_session() -> SparkSession:
    """Returns the singleton SparkSession instance."""
    return SparkInstance().get_spark()
