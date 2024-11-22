from pyspark.sql import SparkSession


_spark_ctx = None

def get_spark():
    '''
    Get or create a spark session
    '''
    global _spark_ctx
    if _spark_ctx is None:
        _spark_ctx = (
            SparkSession.builder.master('local').appName('pipeline')
            .getOrCreate()
        )
    return(_spark_ctx)

spark = get_spark()