hierarchial namespace - Catalog.Schema.Table   and folder structure like /Source/Sales/2025.csv
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
How to transfer Workspace with Parameter Notebook ipynb from one instance to another instance 
Option 1: The Databricks CLI (Recommended for CI/CD)
# 1. Export the notebook from the source workspace
databricks workspace export /Users/your_user/my_notebook.ipynb \
    --target-path ./local_export/my_notebook.ipynb \
    --format IPYNB \
    --profile source_instance
# 2. Import the notebook to the target workspace
databricks workspace import ./local_export/my_notebook.ipynb \
    /Users/your_user/my_notebook.ipynb \
    --language PYTHON \
    --format IPYNB \
    --overwrite \
    --profile target_instance
- Option 2: Databricks Repos (Recommended for Developer Workflow) using GitHub concept
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Enabled UC Unity Catalog - where UC data gets stored 
1. Metadata and Governance Data (The "Catalog") - Catalog and Governance in Databricks Control Plane - Managed by Databricks, secured in Azure
2. Actual Data Files (The "Table Content") - Azure Data Lake Storage Gen2 (ADLS Gen2)
CREATE TABLE prod_catalog.gold.orders
LOCATION 'abfss://data-container@mydatalake.dfs.core.windows.net/prod/orders_delta';
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Liquid clustering - Liquid Clustering is a flexible, dynamic way to organize data files within a Delta Lake table. Instead of pre-defining a rigid structure (like partitioning), it uses a multi-dimensional technique that can adapt to changing query patterns.
Structure - Dynamic, multi-dimensional layout managed by the engine.
Flexibility - High. You can change the clustering columns anytime without rewriting the table structure.
Maintenance - Low. Clustering is automatically maintained during data writes (OPTIMIZE command is still run, but is much more efficient).
CREATE TABLE my_table (
  id INT,
  event_time TIMESTAMP,
  customer_id STRING
)
USING DELTA
CLUSTER BY (customer_id, event_time) -- You define the keys you want to query on
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
core
executor
memory

1. Setting Parameters in Databricks (Cluster Configuration)
# Sets the number of cores used by each executor process (Recommended: 4-8)
spark.executor.cores 4

# Sets the memory allocated to each executor (Recommended: 8GB - 32GB)
# 'm' for MB, 'g' for GB. Leaving headroom for overhead is crucial (see notes below).
spark.executor.memory 12g 

# Sets the memory for the driver (the Master node)
spark.driver.memory 8g

# The default level of parallelism for Shuffle/Join operations
# A good rule of thumb is 2-4 times the total number of cores in the cluster.
# E.g., if you have 10 workers * 4 cores each = 40 total cores. 40 * 4 = 160.
spark.sql.shuffle.partitions 160

# Enables Adaptive Query Execution (AQE), highly recommended for performance
spark.sql.adaptive.enabled true

2. Setting Parameters at Runtime (SparkSession Builder)
That is a vital question for performance tuning! Setting the right Spark parameters for resources like cores and memory is the difference between a job that finishes in minutes and one that crashes or takes hours.

There are three primary ways to set Spark configuration parameters, depending on your environment.

1. Setting Parameters in Databricks (Cluster Configuration)
In an Azure Databricks environment, this is the most common and easiest method, as you set the parameters before the job starts by configuring the cluster.

You use the "Spark Config" box when creating or editing a cluster.

Example Databricks Spark Config Settings
You define the configurations as key-value pairs (one per line):

Properties

# Sets the number of cores used by each executor process (Recommended: 4-8)
spark.executor.cores 4

# Sets the memory allocated to each executor (Recommended: 8GB - 32GB)
# 'm' for MB, 'g' for GB. Leaving headroom for overhead is crucial (see notes below).
spark.executor.memory 12g 

# Sets the memory for the driver (the Master node)
spark.driver.memory 8g

# The default level of parallelism for Shuffle/Join operations
# A good rule of thumb is 2-4 times the total number of cores in the cluster.
# E.g., if you have 10 workers * 4 cores each = 40 total cores. 40 * 4 = 160.
spark.sql.shuffle.partitions 160

# Enables Adaptive Query Execution (AQE), highly recommended for performance
spark.sql.adaptive.enabled true

2. Setting Parameters at Runtime (SparkSession Builder)
If you are running PySpark code outside of Databricks (e.g., locally, in a traditional Spark cluster, or on an ad-hoc submission), you can configure the resources directly when creating the SparkSession.
from pyspark.sql import SparkSession
# Define the number of executors, cores, and memory
SPARK_EXECUTORS = 5
EXECUTOR_CORES = 4
EXECUTOR_MEMORY = "16g"
DRIVER_MEMORY = "8g"

spark = SparkSession.builder \
    .appName("CustomConfigJob") \
    .config("spark.executor.instances", SPARK_EXECUTORS) \
    .config("spark.executor.cores", EXECUTOR_CORES) \
    .config("spark.executor.memory", EXECUTOR_MEMORY) \
    .config("spark.driver.memory", DRIVER_MEMORY) \
    .config("spark.sql.shuffle.partitions", 100) \
    .getOrCreate()

print(f"Spark Session created with {SPARK_EXECUTORS} executors, each having {EXECUTOR_CORES} cores.")

3. Setting Parameters via spark-submit
If you are submitting your Python script (.py) to a cluster via the command line (e.g., Azure HDInsight or a standalone cluster), you use the --conf arguments with spark-submit.
Example spark-submit Command
Bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-cores 5 \
  --executor-memory 18g \
  --driver-memory 4g \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.driver.maxResultSize=4g \
  my_pyspark_script.py

That is a vital question for performance tuning! Setting the right Spark parameters for resources like cores and memory is the difference between a job that finishes in minutes and one that crashes or takes hours.

There are three primary ways to set Spark configuration parameters, depending on your environment.

1. Setting Parameters in Databricks (Cluster Configuration)
In an Azure Databricks environment, this is the most common and easiest method, as you set the parameters before the job starts by configuring the cluster.

You use the "Spark Config" box when creating or editing a cluster.

Example Databricks Spark Config Settings
You define the configurations as key-value pairs (one per line):

Properties

# Sets the number of cores used by each executor process (Recommended: 4-8)
spark.executor.cores 4

# Sets the memory allocated to each executor (Recommended: 8GB - 32GB)
# 'm' for MB, 'g' for GB. Leaving headroom for overhead is crucial (see notes below).
spark.executor.memory 12g 

# Sets the memory for the driver (the Master node)
spark.driver.memory 8g

# The default level of parallelism for Shuffle/Join operations
# A good rule of thumb is 2-4 times the total number of cores in the cluster.
# E.g., if you have 10 workers * 4 cores each = 40 total cores. 40 * 4 = 160.
spark.sql.shuffle.partitions 160

# Enables Adaptive Query Execution (AQE), highly recommended for performance
spark.sql.adaptive.enabled true
2. Setting Parameters at Runtime (SparkSession Builder)
If you are running PySpark code outside of Databricks (e.g., locally, in a traditional Spark cluster, or on an ad-hoc submission), you can configure the resources directly when creating the SparkSession.

Example PySpark Code (Local or Custom Cluster)
Python

from pyspark.sql import SparkSession

# Define the number of executors, cores, and memory
SPARK_EXECUTORS = 5
EXECUTOR_CORES = 4
EXECUTOR_MEMORY = "16g"
DRIVER_MEMORY = "8g"

spark = SparkSession.builder \
    .appName("CustomConfigJob") \
    .config("spark.executor.instances", SPARK_EXECUTORS) \
    .config("spark.executor.cores", EXECUTOR_CORES) \
    .config("spark.executor.memory", EXECUTOR_MEMORY) \
    .config("spark.driver.memory", DRIVER_MEMORY) \
    .config("spark.sql.shuffle.partitions", 100) \
    .getOrCreate()

print(f"Spark Session created with {SPARK_EXECUTORS} executors, each having {EXECUTOR_CORES} cores.")

# Your PySpark code goes here...
3. Setting Parameters via spark-submit
If you are submitting your Python script (.py) to a cluster via the command line (e.g., Azure HDInsight or a standalone cluster), you use the --conf arguments with spark-submit.

Example spark-submit Command
Bash

spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-cores 5 \
  --executor-memory 18g \
  --driver-memory 4g \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.driver.maxResultSize=4g \
  my_pyspark_script.py

4. Crucial Optimization Note: Memory Overhead
When setting memory (spark.executor.memory), always remember that Spark requires Memory Overhead for things like JVM heap, thread stacks, and off-heap memory.
If you request too much memory without accounting for overhead, your job can fail with "Container killed by YARN for exceeding memory limits" errors.
A safe rule of thumb is:
Total Container Memory = spark.executor.memory + spark.executor.memoryOverhead
Spark usually calculates the overhead automatically (typically 10% or at least 384MB), but if you need to manually increase it for memory-intensive operations (like large UDFs):
# Manually setting the overhead to 2GB
spark.executor.memoryOverhead 2048m

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Only Pyspark Join
inner/leftouter/rightouter/fullouter/
leftsemi - similar to exists /
leftanti - similar to not exists /
crossjoin - Cross join
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
df.join(df2, on['key1','key2'], how = 'inner')
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
let us say I have two data files of sizes 5 GB and 5 MB, how to do the performance optimization in this case. - broadcast smaller table to all the worker tables
Option1:
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

# 1. Initialize Spark Session (assuming a running cluster)
# spark = SparkSession.builder.appName("BroadcastJoinOptimization").getOrCreate()

# 2. Load the two DataFrames
# Assume the 5GB file is df_fact and the 5MB file is df_dim
df_fact = spark.read.format("delta").load("path/to/5gb_fact_table")
df_dim = spark.read.format("delta").load("path/to/5mb_dimension_table")

# 3. Perform the Broadcast Join
# The broadcast hint is applied to the smaller (df_dim) DataFrame
joined_df = df_fact.join(
    broadcast(df_dim),
    on=df_fact["key"] == df_dim["key"],
    how="inner"
)

joined_df.explain(mode="cost")
# Look for a "BroadcastHashJoin" operation in the query plan.

# Note: The small DataFrame must be truly small enough to fit in memory on all executors.

Option2:
# Sets the auto-broadcast limit to 52 MB (52,428,800 bytes)
spark.sql.autoBroadcastJoinThreshold 52428800 

### Method 3: Adaptive Query Execution (AQE)

For modern Spark versions (DBR 10.4+), the **Adaptive Query Execution (AQE)** feature often handles this optimization automatically.

* **Configuration:** Ensure this is enabled (it's the default in Databricks):
    ```properties
    spark.sql.adaptive.enabled true
    * **How it helps:** During the query execution, AQE monitors the size of the DataFrames. If it detects that a DataFrame is small enough to be broadcasted (even after filtering, which wasn't known at the start), it dynamically switches the join strategy from a Sort Merge Join to a Broadcast Hash Join.

## Summary of Optimization Steps

1.  **Use the `broadcast(df_dim)` hint (Method 1):** This explicitly tells Spark your intention and ensures the fastest join possible.
2.  **Verify File Format:** Ensure both files are stored in an efficient, splittable format like **Delta Lake or Parquet**. Reading CSVs from a 5 GB file would be significantly slower.
3.  **Ensure Driver/Executor Memory:** Confirm your cluster has enough memory (`spark.driver.memory` and `spark.executor.memory`) to comfortably hold the 5 MB dimension table, plus overhead, on every node.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
let us say I have two data files of sizes 10 GB and 5 GB, how to do the performance optimization in this case.

1. Co-Locate the Data (Partitioning)
both the 10 GB file (df_A) and the 5 GB file (df_B) are partitioned on the same join key and have the same number of partitions,
# Write the 10 GB file partitioned by the join key
df_A.write.partitionBy("key_column").format("delta").mode("overwrite").save("path/to/A")
# Write the 5 GB file partitioned by the exact same key
df_B.write.partitionBy("key_column").format("delta").mode("overwrite").save("path/to/B")

2. Pre-Sort the Data (Z-Ordering / Clustering)
-- In Databricks/Spark SQL:
OPTIMIZE table_A ZORDER BY (key_column);
OPTIMIZE table_B ZORDER BY (key_column);

3. Increase Shuffle Partitions
Rule of Thumb: Set spark.sql.shuffle.partitions to 2x to 4x the total number of CPU cores in your cluster. If you have 10 workers with 8 cores each (80 total cores), aim for 160 to 320 partitions.
spark.sql.shuffle.partitions 320

4. Handle Data Skew (Most Complex)
Action: Use Adaptive Query Execution (AQE) to automatically handle skew.
spark.sql.adaptive.enabled true
spark.sql.adaptive.skewedJoin.enabled true

Summary Strategy
If possible, pre-partition both 10 GB and 5 GB files by the join key.
Enable AQE (spark.sql.adaptive.enabled true).
Increase spark.sql.shuffle.partitions to utilize all cluster cores.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Different Clusters available in Azure Databricks

1. All-Purpose Compute (For Development & Interactive Work)
Cluster Type,Use Case & Strengths,Ideal Users
Standard Cluster,"Interactive Development. Used for writing, debugging, and interactively analyzing data in notebooks.","Data Scientists, Data Analysts, Data Engineers (during development)."
High Concurrency Cluster,Shared & Secure. Optimized for concurrent users and API calls. Required if you need Table Access Control (ACLs) or Unity Catalog security features (like row/column filtering).,"Teams sharing one cluster, production applications requiring maximum security."
Machine Learning Cluster,"Specialized cluster that comes pre-configured with the most popular ML libraries (TensorFlow, PyTorch, Scikit-learn), optimized drivers (like CUDA), and the Databricks ML Runtime.",Machine Learning Engineers and Data Scientists.

2. Jobs Compute (For Production & Automated Workloads)
Cluster Type,Use Case & Strengths,Cost Advantage
Jobs Cluster,"Automated ETL/ELT. Used for scheduled, triggered, or single-run tasks (e.g., daily data ingestion pipeline).",Significantly Cheaper (up to 70% less expensive per DBU) than All-Purpose Compute because it's non-interactive.

3. Serverless Compute (For SQL & Instant Querying)
Cluster Type,Use Case & Strengths,Key Feature
Databricks SQL Warehouse,"BI and Reporting. Optimized specifically for low-latency SQL queries, dashboards, and BI tools (like Power BI).","Serverless & Instant On. No infrastructure management; scales instantly and pauses automatically, making it extremely cost-effective for intermittent querying."

4. SQL Warehouse Sizing
Size,Capacity,Cost/Performance
Small,1 Cluster,Good for development and low concurrency.
Medium,2 Clusters,Standard size for small teams and modest reporting needs.
Large/XL+,4+ Clusters,"Used for high-volume BI, large user groups, or extremely complex queries."

5. Quick Comparison Table
Category,Typical Use Case,Cost Model,Key Feature
All-Purpose,Interactive Notebooks / Development,High DBU Cost,"Manual Start/Stop, Flexibility"
Jobs,Automated ETL/Production Pipelines,Low DBU Cost,Optimized for Unattended Execution
SQL Warehouse,BI / SQL Querying / Dashboards,Serverless DBU Cost,"Instant Scaling, Zero Management"


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Different compute instances available in Azure Databricks
Architect's Guide to Choosing an Instance
When setting up your Databricks cluster, you should choose the VM size based on your primary workload characteristic:

Workload Characteristic,        Recommended VM Series,  Reason
Standard ETL/Data Ingestion,    D-Series (Balanced),    Best value and performance for general processing.
Complex Joins / Large Shuffles, E-Series (High Memory), Prevents out-of-memory errors during data movement.
Heavy UDFs / ML Inferencing,    F-Series (High CPU),    Faster execution of the underlying code logic.
Deep Learning Training,         N-Series (GPU),         Necessary hardware for distributed model training.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Let us say there are multiple pipelines running and the cluster is bleeding, what fixes can be done without doing any code changes

Since the constraints are "no code changes," here are the most effective, non-code fixes you can perform on your clusters and jobs to reduce cost immediately.

1. Cluster Type and Scheduling Adjustments (Most Impactful)
The single biggest cost saver is ensuring you are running the right cluster type for the right job.

A. Switch Production Pipelines to Jobs Compute
If any of your continuous or scheduled production pipelines are running on an All-Purpose Cluster, they are incurring massive cost penalties.

Fix: Immediately switch the job configuration to run on a Jobs Cluster (or a New Job Cluster).

Result: Jobs Compute has a significantly lower DBU rate (up to 70% cheaper) than All-Purpose Compute, making this the fastest way to stop the bleed on production workloads.

B. Aggressive Auto-Termination
Ensure that development and interactive clusters don't sit idle, incurring cost while no one is using them.

Fix: Lower the Auto-Termination threshold on All-Purpose Clusters from the default (e.g., 120 minutes) to a much shorter time (e.g., 30 minutes or even 15 minutes).

Result: Clusters will shut down sooner when idle, stopping the hourly VM charge.

C. Utilize Databricks SQL Warehouses
If any of your "pipelines" involve running scheduled SQL queries, dashboards, or connecting BI tools, they should not be on a general-purpose cluster.

Fix: Switch these workloads to use a Databricks SQL Warehouse (Serverless Compute).

Result: SQL Warehouses have instant auto-scaling and aggressive auto-pausing features, making them extremely cost-effective for intermittent query workloads.

2. Auto-Scaling Optimization
Tuning the cluster's ability to scale up and down efficiently is crucial for cost management.

A. Configure Proper Scaling Range
Review the auto-scaling range (e.g., 2 to 20 workers). If your jobs rarely need 20 workers, they shouldn't be allowed to scale that high.

Fix: Set a tighter Maximum Workers limit based on your job's typical peak load (e.g., set the range to 4 to 8 workers instead of 2 to 20).

Result: Prevents the cluster from unnecessarily scaling out during non-peak hours or when a small job is running.

B. Enable Local Disk Caching (Delta Cache)
Caching frequently accessed data on the local SSD of the worker nodes can drastically reduce I/O cost and execution time (faster jobs cost less DBU time).

Fix: Ensure your cluster is built using an Azure VM series that includes local NVMe SSDs (like the Dsv3 or Esv3 series) and that the Delta Cache is enabled (often enabled by default on newer runtimes).

Result: Data reads are faster, reducing overall cluster runtime without any code change.

3. Concurrency and Scheduling Fixes
Multiple pipelines running concurrently often cause contention and unnecessary cluster wake-ups.

A. Run Dependent Jobs Sequentially
If two pipelines are scheduled at the same time but are dependent, they might wake up two different clusters when only one is needed.

Fix: Update the job scheduler (Azure Data Factory, Databricks Workflow) to ensure dependent pipelines run sequentially on the same cluster or on a shared Jobs cluster.

Result: Better resource utilization and fewer simultaneous cluster wake-ups.

B. Increase Maximum Concurrency
If multiple jobs are assigned to a single Jobs Cluster, they might be waiting in the queue unnecessarily, causing other scheduled jobs to wake up a new cluster.

Fix: If the Jobs Cluster has capacity, increase the Maximum Concurrent Runs setting for that cluster to allow more jobs to run in parallel on the existing hardware instead of spawning new clusters.

C. Utilize Pools
If your jobs use multiple small clusters that start and stop frequently, the start-up time adds to the cost.

Fix: Configure a Pool of idle VM instances. Configure your Jobs Clusters to pull workers from this pre-warmed pool.

Result: Cluster start time drops from minutes to seconds. You only pay for the idle VM time (which is less than an active cluster), and the faster spin-up time gets the job running and finished faster.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Where to check the logs of data bricks and how to read logs for performance improvement
Graphana
Ganglia
Azure Monitor using Jar files
Log Analytics is the base
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
https://www.kleeto.in/loginnew.php
Username : biradar.shiv88@gmail.com
Password : -g25IWdf


