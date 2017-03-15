### Execution process
* Any Spark application spins off a single driver process (that can contain multiple jobs) on the master node that then directs executor processes (that contain multiple tasks) distributed to a number of worker nodes.
* The driver process determines the number and the composition of the task processes directed to the executor nodes based on the graph generated for the given job. Note, that any worker node can execute tasks from a number of different jobs.

![execution process](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_01_02.jpg)

* A Spark job is associated with a chain of object dependencies organized in a direct acyclic graph (DAG) such as the following example generated from the Spark UI. Given this, Spark can optimize the scheduling (for example, determine the number of tasks and workers required) and execution of these tasks.

![dag](https://www.safaribooksonline.com/library/view/learning-pyspark/9781786463708/graphics/B05793_01_03.jpg)