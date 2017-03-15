### Execution process
* Any Spark application spins off a single driver process (that can contain multiple jobs) on the master node that then directs executor processes (that contain multiple tasks) distributed to a number of worker nodes.
* The driver process determines the number and the composition of the task processes directed to the executor nodes based on the graph generated for the given job. Note, that any worker node can execute tasks from a number of different jobs.

![execution process](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_01_02.jpg)

* A Spark job is associated with a chain of object dependencies organized in a direct acyclic graph (DAG) such as the following example generated from the Spark UI. Given this, Spark can optimize the scheduling (for example, determine the number of tasks and workers required) and execution of these tasks.

![dag](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_01_03.jpg)

### Resilient Distributed Dataset
* Apache Spark is built around a distributed collection of immutable Java Virtual Machine (JVM) objects called Resilient Distributed Datasets (RDDs for short). 
* As we are working with Python, it is important to note that the Python data is stored within these JVM objects. 
* These objects allow any job to perform calculations very quickly. RDDs are calculated against, cached, and stored in-memory: a scheme that results in orders of magnitude faster computations compared to other traditional distributed frameworks like Apache Hadoop.
* RDDs expose some coarse-grained transformations (such as map(...), reduce(...), and filter(...) which we will cover in greater detail in Chapter 2, Resilient Distributed Datasets), keeping the flexibility and extensibility of the Hadoop platform to perform a wide variety of calculations. 
*RDDs apply and log transformations to the data in parallel, resulting in both increased speed and fault-tolerance. By registering the transformations, RDDs provide data lineage - a form of an ancestry tree for each intermediate step in the form of a graph. This, in effect, guards the RDDs against data loss - if a partition of an RDD is lost it still has enough information to recreate that partition instead of simply depending on replication.