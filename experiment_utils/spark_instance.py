"""
Spark instance
"""

from pyspark.sql import SparkSession
from typing import Optional


class SparkInstance:
    """Singleton class to create a Spark session"""
    _instance: Optional['SparkInstance'] = None
    spark: SparkSession = None

    def __new__(cls) -> 'SparkInstance':
        if cls._instance is None:
            cls._instance = super(SparkInstance, cls).__new__(cls)
            cls._instance.spark = (
                SparkSession.builder.master('local').appName('experiment_utils')
                .getOrCreate()
            )
        return cls._instance

    def get_spark(self) -> SparkSession:
        """Get the Spark session"""
        return self.spark


spark_instance = SparkInstance()
spark = spark_instance.get_spark()
