Learning path = Build machine learning solutions using Azure Databricks -- https://learn.microsoft.com/en-us/training/paths/build-operate-machine-learning-solutions-azure-databricks/
-------------------------------------------------------------------------------------------------------------------
Module1 = Explore Azure Databricks -- https://learn.microsoft.com/en-us/training/modules/explore-azure-databricks/

A data lakehouse is a data management approach that blends the strengths of both data lakes and data warehouses. It offers scalable storage and processing, allowing organizations to handle diverse workloads—such as machine learning and business intelligence—without relying on separate, disconnected systems. By centralizing data, a lakehouse supports a single source of truth, reduces duplicate costs, and ensures that information stays up to date.

Many lakehouses follow a layered design pattern where data is gradually improved, enriched, and refined as it moves through different stages of processing. This layered approach—commonly called the medallion architecture—organizes data into stages that build on one another, making it easier to manage and use effectively.

The Databricks lakehouse uses two key technologies:

Delta Lake: an optimized storage layer that supports ACID transactions and schema enforcement.
Unity Catalog: a unified, fine-grained governance solution for data and AI.

-- Identify Azure Databricks workloads
The platform facilitates data exploration, visualization, and the development of data pipelines.
Supports various ML frameworks such as TensorFlow, PyTorch, and Scikit-learn, making it versatile for different ML tasks.

Workspaces are tied to Unity Catalog (when enabled) for centralized data governance, ensuring secure access to data across the organization. Each workspace is also linked to an underlying Azure resource group (including a managed resource group) that holds the compute, networking, and storage resources Databricks uses behind the scenes.

Allows users to switch between languages within the same notebook using magic commands. This flexibility makes notebooks well-suited for exploratory data analysis, data visualization, machine learning experiments, and building complex data pipelines.

-- Data governance using Unity Catalog and Microsoft Purview
Unity Catalog - provides a centralized way to manage access, discovery, lineage, audit logs, and quality monitoring across data and AI assets within Azure Databricks. It applies consistently across all workspaces in a region.
Unity Catalog organizes data assets using a structured three-level hierarchy: catalog.schema.table_or_other_object

Microsoft Purview - is a data governance service that lets you manage and oversee data across on-premises systems, multiple clouds, and SaaS platforms. It includes features such as data discovery, classification, lineage tracking, and access governance.
When Microsoft Purview is integrated with Azure Databricks and Unity Catalog, Purview can discover Lakehouse data and ingest its metadata into the Data Map. This allows you to apply consistent governance across your entire data environment, while acting as a central catalog that brings together metadata from different sources.
Microsoft Purview can scan the workspace-level Hive metastore in Azure Databricks.

-------------------------------------------------------------------------------------------------------------------
Module2 = Use Apache Spark in Azure Databricks
Azure Databricks is built on Apache Spark and enables data engineers and analysts to run Spark jobs to transform, analyze and visualize data at scale.

Apache Spark is an open-source, distributed computing system that processes large amounts of data across many machines while keeping as much as possible in memory (RAM). This design choice makes Spark both fast and scalable, capable of handling anything from small datasets on your laptop to petabytes of data on massive clusters.

With Spark, you don’t need separate systems for real-time streaming, batch processing, SQL queries, or machine learning. Everything can be done in Spark with a consistent set of APIs.

Spark Core -- provides the foundation for distributed computing, including task scheduling, memory management, and fault recovery.
Spark SQL -- lets you work with structured data using a language most analysts already know: SQL. It also integrates with external data sources like Hive, Parquet, and JSON.
Spark Streaming -- allows you to process data in near real time, which is useful for applications like fraud detection or monitoring system logs.
MLlib -- is Spark’s machine learning library. It provides scalable implementations of algorithms for classification, clustering, regression, and recommendation.
GraphX -- is used for graph computation, such as analyzing social networks or modeling relationships between entities.

Lazy Evaluation = Spark simply records these operations in a plan. Nothing is actually computed yet. This approach allows Spark to look at the full sequence of operations before deciding the most efficient way to execute them.
DAG - Directed Acyclic Graph (DAG)

When you need more power, you can scale out by running Spark on a standalone cluster of machines, or integrate it with resource managers such as Hadoop YARN or Apache Mesos, which handle scheduling and resource allocation.

Spark Cluster = Azure Databricks Workspace 
Azure Databricks Workspace = Control Plane + Compute Plane
	Control Plane = Web Application + Compute Orchestration + Unity Catalog + Queries and Code
	Compute Plane = Multiple Compute instances
		Serverless
		Classic Compute Plane
		
The Databricks appliance is deployed into Azure as a managed resource group within your subscription. This resource group contains the driver and worker VMs for your clusters, along with other required resources, including a virtual network, a security group, and a storage account. All metadata for your cluster, such as scheduled jobs, is stored in an Azure Database with geo-replication for fault tolerance.

Every Databricks workspace has a storage account in your subscription that holds system data (notebooks, logs, job metadata), the distributed file system (DBFS), and catalog assets (if you have Unity Catalog enabled), with additional controls for networking, firewalling, and access to ensure security and proper isolation.

Using Databricks AI Assistant
You interact with the assistant in two main ways:
	Natural language prompts—you can type plain English instructions in the chat-like interface, and it will insert code into your notebook.
	Slash commands—quick commands such as /explain, /fix, or /optimize that let you act on selected code. For example, 
		/explain breaks down a complex function into simpler steps, 
		/fix can attempt to resolve syntax or runtime errors, and 
		/optimize suggests performance improvements such as repartitioning or using efficient Spark functions.

After setting up a notebook and attaching it to a cluster, you can use Spark to read and process data files. Spark supports a wide range of formats—such as CSV, JSON, Parquet, ORC, Avro, and Delta—and Databricks provides built-in connectors to access files stored in the workspace, in Azure Data Lake or Blob Storage, or in other external systems.

/Workspace/MicrosoftLearnings

-------------------------------------------------------------------------------------------------------------------
Module3 = Build machine learning solutions using Azure Databricks

To train a predictive model, you use a machine learning framework to determine a relationship between the features of entities, and the labels you want to predict for them. For example, you might train a model to predict the expected price of a home based on features such as the property size, number of bedrooms, postal code, and so on.

Azure Databricks provides an Apache Spark based data processing platform that supports multiple popular machine learning frameworks; including Scikit-Learn, PyTorch, TensorFlow, and others. 
This module uses the Spark MLlib machine learning framework to show examples, but the principles it describes apply to all machine learning frameworks.

At a simplistic level, a machine learning model is a function that takes the features of an observed entity (its characteristics) and performs a calculation on them to return a predicted label. It''s common to refer in general to the features as x and the predicted label as y; so in effect, a machine learning model is the function f in the expression y = f(x).

Supervised machine learning in which the model is trained using data that includes known label values (so an algorithm uses the existing data to establish a relationship between x and y, resulting in a function that can be applied to x to calculate y). Regression Classification
Unsupervised machine learning in which the model is trained using only feature (x) values, and groups (or clusters) observations with similar features.

Unsupervised machine learning
The most common form of unsupervised learning is clustering, in which the features of data cases are considered as a vector of points in multidimensional space. The goal of a clustering algorithm is to define clusters that group the points so that cases with similar features are close together, but the clusters are clearly separated from one another.

If you are going to be implementing machine learning solutions, create a cluster with one of the ML runtimes. You can choose a CPU-based runtime for classical machine learning scenarios, or a GPU-based runtime if you need to build complex neural networks with deep learning frameworks, which can take advantage of a GPUs ability to efficiently process matrix and vector based data.

Azure Databricks is built on Apache Spark, a highly scalable platform for distributed data processing.

-- Prepare data for machine learning
Typically, the preparation of data involves two key tasks:
Data cleansing: Identifying and mitigating issues in the data that will affect its usefulness for machine learning.
Feature engineering and pre-processing: Selecting and transforming suitable features for model training.

--Data cleansing
Incomplete data + Errors + Outliers + Incorrect data types + Unbalanced data

--For example, the following code loads data from a text file into a dataframe:
df = spark.read.format("csv").option("header", "true").load("/myfolder/mydata.csv")
--Alternatively, if the data has been loaded into a delta table in the Azure Databricks workspace, you can use a SQL query to load its data into a dataframe:
df = spark.sql("SELECT * FROM mytable")
--dropna method to remove any rows that include null values, and assigns specific data types to columns in the dataframe.
clean_data = df.dropna().select(col("column1").astype("string"),
                                col("column2").astype("float"))

-- Feature engineering and preprocessing - https://learn.microsoft.com/en-us/training/modules/machine-learning-azure-databricks/4-prepare-data-for-machine-learning
Deriving new features: suppose a dataset includes a date column and you suspect that the complete date may not be an important predictive factor in identifying the label, but that the day of the week might be. You could create a new day_of_week feature derived from the date and test your theory.

Discretizing numeric features: In some cases, a numeric value might prove more predictive when discretized into categories that represent specific ranges of values. For example, you might take the numeric values in a price feature and assign them into low, medium, and high categories based on appropriate thresholds.

Encoding categorical features: Many datasets include categorical data that is represented by string values. However, most machine learning algorithms work best with numeric data. It is therefore common to assign numeric codes to represent categories instead of strings. For example, a dataset of product details might include a feature for color that can have a value of "Green", "Red", or "Blue". You could encode these values using simple integer codes such as 0 for "Green", 1 for "Red", and 2 for "Blue". Alternatively, you could use a one-hot encoding technique in which you create a new column for each possible category, and assign the value 1 or 0 to each column as appropriate for each row, like this:
Original Color Column	Green	Red	Blue
Green                       1	0	0
Blue                    	0	0	1
Red	                        0	1	0

Scaling (normalizing) numeric values: Numerical data values are often on different scales or units of measurement from one another. Machine learning algorithms process them all as absolute numeric values, and features with larger values can often dominate the training of the model. To resolve this problem, it is common to scale all of the numeric columns so that the individual values for a single column maintain the same proportional relationship to one another, but all of the numeric columns are on a similar scale. For example, suppose a dataset contains length and weight values measured in meters and kilograms. You could convert both of these features to a scaled value between 0 and 1 like this:

In Spark MLLib, you can chain a sequence of evaluators and transformers together in a pipeline that performs all the feature engineering and preprocessing steps you need to prepare your data. The pipeline can end with a machine learning algorithm that acts as an evaluator to determine the operations required to predict a label from the prepared features. The output of the pipeline is a machine learning model, which is in fact a transformer that can be used to apply the model function to features in a dataframe and predict the corresponding label values.

-- Train a machine learning model
Models that predict well for the data on which they were trained but which don''t work well with new data are described as overfitted to the training data. Typically, you should train the model with around 70% of the data and hold back around 30% for validation.

Machine learning algorithms
- Logistic regression algorithms that iteratively apply logistic functions to calculate a value between 0 and 1 that represents the probability for each possible class, and optimize the function''s coefficients based on the differences between the predicted class and the actual known label value.
- Tree-based functions that define a decision tree in which an individual feature is considered; and based on its value, another feature is considered, and so on, until an appropriate class label is determined.
- Ensemble algorithms that combine multiple techniques to find the optimal overall predictive function.
The "best" algorithm depends on your data, and usually requires iterative trial and error to determine.

Hyperparameters - enable you to control things like the level of randomness you want to allow in the model (so it generalizes well but still produces acceptably accurate predictions), the number of iterations performed to find an optimal model (enabling you to avoid overfitting and optimize training time), the number of branches considered in a tree model, and other algorithm-specific factors.

-- Fitting a model
The following example shows the code used to initiate training of a logistic regression model using the Spark MLlib framework. The training data is provided as a dataframe in which the labels are in a column of integer values, and the corresponding features are represented as a single vector (array) of values. In this example, two hyperparameters (maxIter and regParam) have also been specified.

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3)
model = lr.fit(training_df)

-- Evaluating regression models
Mean Squared Error (MSE): This metric is calculated by squaring the differences between each prediction and actual value, and adding the squared differences together, and calculating the mean (average). Squaring the values makes the differences absolute (ignoring whether the difference is negative or positive) and gives more weight to larger differences.

Root Mean Squared Error (RMSE): While the MSE metric is a good indication of the level of error in the models predictions, it doesn''t relate to the actual unit of measurement of the label. For example, in a model that predicts sales (in US dollars), the MSE value actually represents the dollar values squared. To evaluate how far off the predictions are in terms of dollars, you need to calculate the square root of the MSE.

Coefficient of Determination (R2): The R2 metric measures the correlation between the squared feature and predicted values. This results in a value between 0 and 1 that measures the amount of variance that can be explained by the model. The closer this value is to 1, the better the model predicts.
R-squared, is a statistical measure that represents the proportion of the variance for a dependent variable that''s explained by an independent variable or variables in a regression model.
R2 =1: The model perfectly explains all the variability of the dependent variable. In a scatter plot, all the data points would fall perfectly on the regression line. This is rare in real-world data.
R2 =0: The model explains none of the variability of the dependent variable. The regression line is not a better predictor than the mean of the dependent variable.
0<R2<1: The model explains a certain proportion of the variance. For example, if R2 =0.75, it means that 75% of the variation in the dependent variable is predictable from the independent variable(s).


-------------------------------------------------------------------------------------------------------------------
Module4 = Use MLflow in Azure Databricks

MLflow is an open source platform for managing the machine learning lifecycle that is natively supported in Azure Databricks.

-- Capabilities of MLflow
There are four components to MLflow:
- MLflow Tracking 
	- MLflow Tracking allows data scientists to work with experiments in which they process and analyze data or train machine learning models. For each run in an experiment, a data scientist can log parameter values, versions of libraries used, model evaluation metrics, and generated output files; including images of data visualizations and model files.
- MLflow Projects 
	- An MLflow Project is a way of packaging up code for consistent deployment and reproducibility of results. MLflow supports several environments for projects, including the use of Conda and Docker to define consistent Python code execution environments.
- MLflow Models
	- MLflow offers a standardized format for packaging models for distribution. This standardized model format allows MLflow to work with models generated from several popular libraries, including Scikit-Learn, PyTorch, MLlib, and others.
- MLflow Model Registry
	- The MLflow Model Registry allows data scientists to register trained models. MLflow Models and MLflow Projects use the MLflow Model Registry to enable machine learning engineers to deploy and serve models for client applications to consume.

-- Run experiments with MLflow
MLflow experiments allow data scientists to track training runs in a collection called an experiment. Experiment runs are useful for comparing changes over time or comparing the relative performance of models with different hyperparameter values.

with mlflow.start_run():
    mlflow.log_param("input1", input1)
    mlflow.log_param("input2", input2)
    # Perform operations here like model training.
    mlflow.log_metric("rmse", rmse)

In this case, the experiment''s name is the name of the notebook. It''s possible to export a variable named MLFLOW_EXPERIMENT_NAME to change the name of your experiment should you choose.

-- Register and serve models with MLflow
- Registering a model allows you to serve the model for real-time, streaming, or batch inferencing. Registration makes the process of using a trained model easy, as now data scientists don't need to develop application code; the serving process builds that wrapper and exposes a REST API or method for batch scoring automatically.
- Registering a model allows you to create new versions of that model over time; giving you the opportunity to track model changes and even perform comparisons between different historical versions of models.





















when i run the following 

getting the following error




