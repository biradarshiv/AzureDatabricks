# Real-Time-Streaming-with-Azure-Databricks

## Project Overview
Welcome to the "Real-Time Streaming with Azure Databricks" repository. This project demonstrates an end-to-end solution for real-time data streaming and analysis using Azure Databricks and Azure Event Hubs, with visualization in Power BI. It's an in-depth guide covering the setup, configuration, and implementation of a streaming data pipeline following the medallion architecture.

## Getting Started
To get started with this project, clone the repository and follow the guidance provided in this YouTube tutorial.
https://www.youtube.com/watch?v=pwWIegHgNRw

🔗 Spark Structured Streaming API: [https://spark.apache.org/docs/latest/...](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ss/index.html)
🔗 Watermarking: [https://www.databricks.com/blog/featu...](https://www.databricks.com/blog/feature-deep-dive-watermarking-apache-spark-structured-streaming)

## Repository Contents
- `Real-time Data Processing with Azure Databricks (and Event Hubs).ipynb`: The Databricks notebook used for data processing at each layer of the medallion architecture.
- `data.txt`: Contains sample data and JSON structures for streaming simulation.
- `Azure Solution Architecture.png`: High level solution architecture.

## Prerequisites
- Active Azure subscription with access to Azure Databricks and Event Hubs.
- Databricks Workspace with Unity Catalog Enabled.
- Azure Event Hubs Service.
- Power BI Desktop (Windows).
- Familiarity with Python, Spark, SQL, and basic data engineering concepts.

## STEPS DONE
Databricks > Compute > Open a Compute Cluster/Instance > Libraries > Install New > Maven > Specific Packages > select "Maven Central" > search eventshub-spark_2.12 > select > install 

EventHubNameSpace > EventHub > Shared access Policies > Policy Name = 'databricks' > listen > create
Now click on policy databricks, created in above step > Copy "Connection string-primay key" > assign this value to variable "connectionString" in the ipynb/notebook file
variable "connectionString" = eh-demo

Open event hub "eh-demo" > Features > Generate Data (Preview) > Dataset = User Defined Payload > Content Type = JSON > Click Send

To open the file system click on Catalog > Browse DBFS (this is a top right corner button) > if not visible then Click on Your User name > Admin Settings > Workspace Settings > Advnaced > DBFS File Server > this should be enabled > Refresh the page



