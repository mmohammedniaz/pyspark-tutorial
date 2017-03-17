There are two ways to create an RDD in PySpark: you can either .parallelize(...) a collection (list or an array of some elements):
>    data = sc.parallelize([('Amber', 22), ('Alfred', 23), ('Skye',4), ('Albert', 12), ('Amber', 9)])

Object holding data type - **ParallelCollectionRDD**

Or you can reference a file (or files) located either locally or somewhere externally:
>    data_from_file = sc. textFile('/Users/drabast/Documents/VS14MORT.txt.gz',4)

Object holding data type - **MapPartitionsRDD**
The last parameter in sc.textFile(..., n) specifies the number of partitions the dataset is divided into.
