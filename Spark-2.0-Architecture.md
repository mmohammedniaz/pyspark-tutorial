![spark-arch](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_01_05.jpg)

The three overriding themes of the Apache Spark 2.0 release surround performance enhancements (via Tungsten Phase 2), the introduction of structured streaming, and unifying Datasets and DataFrames.

## Unifying Datasets and DataFrames
The goal for datasets was to provide a type-safe, programming interface. This allowed developers to work with semi-structured data (like JSON or key-value pairs) with compile time type safety (that is, production applications can be checked for errors before they run). Part of the reason why Python does not implement a Dataset API is because Python is not a type-safe language.

![datasets & dataframes](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_01_06.jpg)


The Dataset API provides a type-safe, object-oriented programming interface. Datasets can take advantage of the Catalyst optimizer by exposing expressions and data fields to the query planner and Project Tungsten's Fast In-memory encoding. But with DataFrame and Dataset now unified as part of Apache Spark 2.0, DataFrame is now an alias for the Dataset Untyped API. More specifically:

> DataFrame = Dataset[Row]

![unified](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_01_07.jpg)

## Introducing SparkSession
* Older ways - SparkConf, SparkContext, SQLContext, and HiveContext to execute your various Spark queries for configuration, Spark context, SQL context, and Hive context respectively.
* The SparkSession is essentially the combination of these contexts including StreamingContext.

For example, instead of writing:

>   df = sqlContext.read.format('json').load('py/test/sql/people.json')

New way,
 
>   df = spark.read.format('json').load('py/test/sql/people.json')
>   df = spark.read.json('py/test/sql/people.json')

**The SparkSession is now the entry point for reading data, working with metadata, configuring the session, and managing the cluster resources.**