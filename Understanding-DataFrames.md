A DataFrame is an immutable distributed collection of data that is organized into named columns analogous to a table in a relational database. Introduced as an experimental feature within Apache Spark 1.0 as SchemaRDD, they were renamed to DataFrames as part of the Apache Spark 1.3 release. For readers who are familiar with Python Pandas DataFrame or R DataFrame, a Spark DataFrame is a similar concept in that it allows users to easily work with structured data (for example, data tables); there are some differences as well so please temper your expectations.

By imposing a structure onto a distributed collection of data, this allows Spark users to query structured data in Spark SQL or using expression methods (instead of lambdas)

![df](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_03_02.jpg)

### Performance

![performance](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_03_03.jpg)
