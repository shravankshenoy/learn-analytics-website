## Introduction
Spark is a framework (engine) for large scale data processing. Pyspark helps users use Spark functionalities using Python
\
\
Two things which make Spark powerful are:
* **Distributed computing** : Large computational tasks are divided into smaller, independent sub-tasks and executed in parallel (eg. Instead of filtering `region = East` in a 10GB dataset, we divide the data into 10 1 GB subdataset, send each subdataset to a separate machine, filter `region = East` parallelly on all the ten subdatasets and then combine the filtered results from each machine)
* **Lazy evaluation** :  Feature in Spark wherein a transformation is not executed immediately but only much later (more details later)

## Distributed Computing and Spark Architecture
For distributed computing, we essentially need to have multiple machines (could be real computer or a virtual machine). Also when there are multiple machines, we need one more machine to assign the tasks to different machines. Based on this logic we introduce following terms:
* **Node** : A machine
* **Worker Node** : The machine which performs the computation
* **Executor** : A Java process (process = running program) running on a worker node that performs computations. A single worker node can have multiple executors
* **Core** : A single executor has multiple "cores." These cores represent the computational units or threads available within that Executor to run tasks concurrently
* **Driver Node** : Analyzes the program given by user and then distributes the work amongst the executors (hence the central coordinating component)
* **Cluster manager** : Manages CPU and memory of worker nodes based on instruction received from driver node

* Memory vs Disk : 
    * Memory = The RAM (Random Access Memory) available on your Spark executors (i.e., the worker nodes) 
    * Disk = The local hard drive (or SSD) of those executors.

When we cache data, it is stored in memory of if too large spills to disk (more info later)
Suppose number of executor-cores is 4 and num-executors is 3, then spark can run 12 tasks in parallel. Also a cluster manager does nothing more to Apache Spark, but offering resources, and once Spark executors launch, they directly communicate with the driver to run tasks.
\
Below is a picture which shows the Spark architecture
\
![Spark Architecture](spark-architecture-1.png)
\
Since we have multiple nodes, we also need to split the data into smaller chunks, so that the data can be processed in parallel. These smaller chunks are called partitions.

* **Dataset** : Data that we process, generally split into multiple smaller chunks
* **Partition** : Partition is an atomic chunk of data which is subset of the overall dataset 

For a quick video understanding of the Spark architecture refer : [DE Zoomcamp 5.4.1 - Anatomy of a Spark Cluster](https://www.youtube.com/watch?v=68CipcZt7ZA)

* **Shuffling** : Redistributing data amongst partitions. For operations like group by, join, repartition. 

Below is an example of shuffling when we do group by. During shuffling data moves over network as it has to move from one node to another node (and nodes are connected to each other via a network). Shuffling can happen b/w executors on same node (no network transfer) or different nodes (network transfer). In the picture below, each small box represents one row of data, each large box represents a partition

![Shuffling during groupby](shuffle-group-by.png)

* **Repartition** : A transformation that reshuffles data into fixed number of partitions (given by user eg. `df_repartitioned = df.repartition(10)`). Each resulting partition will be of similar size. Reason to use repartition include:
    * Improving parallelism : Leverage all cores (eg. you have 8 cores but only 1 partition, then 7 cores will be idle) 
    * Balance data and avoiding skewed partition (eg. if one partition is much bigger than other partitions, then that core will take much longer time than others or will have OOM issue)

* **Coalesce** : A transformation used to reduce the number of partitions by clubbing data from multiple partitions into single partition

* **Repartitions vs Coalesce** : Unlike repartition, coalesce avoids full shuffle and only moves data from the extra nodes. Also repartition can increase or reduce the number of partitions whereas coalesce can only reduce the number of partitions
```
# Before coalesce (each number represents a row number)
Node 1 = 1,2
Node 2 = 3,4,5
Node 3 = 6,7,8,9
Node 4 = 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24

# After repartition (into 4 partitions)
Node 1 = 1, 4, 5, 8, 20, 21
Node 2 = 2, 3, 7, 10,11, 12
Node 3 = 6, 13, 15, 17, 18, 19
Node 4 = 9, 14, 16, 22, 23, 24

# After coalesce
Node 1 = 1,2,3,4,5,6,7,8,9  (partitions from node 2 and 3 moved to 1)
Node 4 = 10,11,12,13,14,15,16,17,18,19,20,21,22,23,24

```

* **Broadcasting** : Sending a copy of a dataset (typically small) to all the executors. This is an optimization used during join when one of the two datasets is small enough to fit in memory. THe smaller dataset can join with partitions of the larger dataset locally. By broadcasting, you avoid the shuffle and speed up the join.

```
Let’s say:
Large dataset: orders (100 million rows)
Small dataset: countries (200 rows)

A broadcast join:
1. Sends the countries dataset to all worker nodes.
2. Each executor joins its partition of orders with the broadcasted countries locally.

Hence no shuffling needed.
```

* When we write data to an object store, Spark converts logical partitions to physical files. Hence we will have 1 file corresponding to each partition (unless we do bucketing)

```
# In Spark memory 
Executor 1: Partition 0 
Executor 2: Partition 1

# On doing df.write.parquet("s3://my-bucket/output/") below is how data is stored in object store like AWS S3 bucket 

├── part-00000-xxxx.snappy.parquet
├── part-00001-xxxx.snappy.parquet

```

### Lazy Evaluation and DAG
Spark doesn’t run your code immediately when you define transformations — instead, it builds a logical plan and only executes it when you call an action. First lets define transformation and action

* **Transformation** : An operation which transforms the dataframe in some way, but does not perform the transformation as soon as we run the function. Instead the step is stored for future execution (in a logical plan). Includes functions like filter(), groupBy(), union(), join(), select(), map()
* **Action** : An operation that triggers the execution of the computations defined by preceding transformations. This will returns a result to the driver or writes data to an external storage system. Includes functions like count(), collect(),first(), show(), write(), reduce()
* **DAG** : An optimized version of logical plan which stores each step as a node of a graph

```
df = spark.read.csv("data.csv")  # No data read yet
df2 = df.filter(df.age > 30)     # Still no filtering yet
df3 = df2.select("name", "age")  # Still nothing executed

df3.show()  # Action → now Spark reads, filters, and selects in one go

## DAG
CSV Scan  -->  Filter (age > 30)  -->  Select (name, age)  -->  Collect/Show

```
* Transformations are of 2 types:
  * **Narrow transformation** : No shuffle required, because output partition depends only on the corresponding input partition (because we can operate on each row independently)(eg. filter(), select(), map(), union())
  * **Wide transformation** : Shuffle required, because output partitions depends on multiple input partitions (eg. groupby(), join())

* In Spark, execution of program is broken down into job, stage and task

* **Job** : A job is triggered when an action is called
* **Stage** : Each job is split into stages based on shuffle boundaries i.e. when we encounter a wide transformation
* **Task** : Smallest unit of work in Spark. Each task processes one partition of data. All tasks in the same stage do the same set of operations, just on different partitions

Below is an example which puts all of this together

```
#### Program
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

df = spark.read.csv("people.csv", header=True, inferSchema=True) # narrow transformation
result = df.filter(df.age > 30) \ # narrow transformation
           .groupBy("city") \ # wide transformation 
           .count() # wide transformation (because in conjunction with groupby)

result.show() # action

#### Execution
1. Job
The .show() is an action → Spark starts Job 1.

2. Stages
Stage 0 : 
 - Reads CSV from storage
 - Filters age > 30 (narrow transformations)
 - Prepares data for shuffling based on city
Stage 1 :
 - Reads shuffled data
 - Performs groupBy("city") and count() (wide transformations)
 - Sends results to driver for .show()

3. Tasks
If the CSV had 4 partitions initially:
Stage 0 → 4 tasks (each task processes one partition of input data)
Stage 1 → number of shuffle partitions (default 200 unless changed with spark.sql.shuffle.partitions), so 200 tasks by default.


#### DAG diagram
Job 1 (triggered by .show())

Stage 0 (Filter Stage - Narrow)
    Task 0 → Partition 0
    Task 1 → Partition 1
    Task 2 → Partition 2
    Task 3 → Partition 3
        ↓
[Shuffle occurs here based on 'city']
        ↓
Stage 1 (Aggregation Stage - Wide)
    Task 0 → Shuffle Partition 0
    Task 1 → Shuffle Partition 1
    ...
    Task 199 → Shuffle Partition 199


```

### Miscellaneous
When a program is loaded into memory and begins to run, it becomes a process. A process involves not only the program's instructions but also its current state, including the values of its variables, the program counter (which indicates the next instruction to be executed), and the resources it utilizes, such as CPU time, memory, and I/O devices. A single program can give rise to multiple processes, each running independently. For example, opening two separate instances of a web browser (the program) creates two distinct processes. 
Think of Spark as a kitchen:

Worker Node = Kitchen building

Executor = Chef team

Core = Individual cook

The more cooks (cores), the more dishes (tasks) can be prepared simultaneously.

* Bucketing: Bucketing is a physical layout on disk, not an in-memory execution strategy.
* An out-of-memory (OOM) error when processing large partitions in Spark often indicates that the data within a partition exceeds the available memory of the executor or driver

When you use partitioning + bucketing:
Data is first partitioned by a column (e.g., country)
Then, within each partition, data is bucketed by another column (e.g., user_id)
```
df.write \
  .partitionBy("country") \
  .bucketBy(4, "user_id") \
  .sortBy("user_id") \
  .format("parquet") \
  .saveAsTable("user_data")

/table/
├── country=US/
│   ├── bucket_00000
│   ├── bucket_00001
│   └── ...
├── country=IN/
│   ├── bucket_00000
│   ├── bucket_00001
│   └── ...
```
Thus each partition has its set of buckets
Partitioning helps with query filtering (e.g., WHERE country = 'IN') — avoids scanning unnecessary folders.
Bucketing helps with joins or aggregations by ensuring data is hashed and grouped efficiently within partitions.
.collect() brings all records from all partitions across the cluster to the driver node as a Python list.

1. Where Are Spark Partitions Stored?
It depends on the stage of execution:

🧠 In Memory (during computation)
While your DataFrame or RDD is being processed, each partition lives in memory (RAM) on a Spark executor (running on a worker node).

If the partition is too large for memory and you’ve enabled spilling, it may spill to disk on that worker node.

Caching/persisting also stores partitions in memory (or memory + disk).

So during execution, partitions are temporary blocks of data held by executors across the cluster.
What Happens When You Write Data to an Object Store?
Each partition becomes a physical file (or group of files) in the object store:
Spark converts logical partitions to physical files (like .snappy.parquet or .csv) on S3, ADLS, GCS, etc.

One file per partition (unless you're using bucketing or coalescing).

Example file names:

DAG Stage : A stage is a set of tasks that can be executed without requiring data shuffling across the cluster. Stages are separated by shuffle boundaries — points where data must be redistributed (e.g., for groupBy, join, reduceByKey).Inside a stage, all transformations are narrow (data stays within the same partition). Between stages, Spark inserts a shuffle (wide transformation).

Actually logical plan is first optimized and converted to physical plan, and then that is transformed to DAG. 

```
df = spark.read.csv("data.csv", header=True, inferSchema=True)
df2 = df.filter(df.age > 30)             # Narrow transformation
df3 = df2.select("name", "age")          # Narrow transformation
df4 = df3.groupBy("age").count()         # Wide transformation → needs shuffle
df4.show()          

(Stage 0)    Read CSV → Filter → Select
                  |  shuffle (by age)
                  v
(Stage 1)    Shuffle Read → GroupBy(age) → Count → Show

```
Note : groupBy("city") needs all rows with the same city key to be in the same partition



1. https://www.reddit.com/r/dataengineering/comments/15hosx6/relation_between_no_of_executors_and_cores/
2. https://www.youtube.com/watch?v=ffHboqNoW_A
3. https://stackoverflow.com/questions/31610971/spark-repartition-vs-coalesce
4. https://stackoverflow.com/questions/34722415/understand-spark-cluster-manager-master-and-driver-nodes