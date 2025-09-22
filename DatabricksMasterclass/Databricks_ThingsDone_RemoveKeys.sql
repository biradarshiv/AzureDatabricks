Databricks is cloud independent - All 3 Major cloud providers can get started with the databricks (AWS, Azure, GCP)
Databricks with spark clusters is the way to go
What is the spark cluster - cluster is a group of machines - each machine is being called as a node - 1 Node = 1 Machine - 
Every cluster has a cluster manager
One machine or one node is used as a "Driver Program" with "spark context"
Worker node1 = Executor(Task + Task) + Cache 
Worker node2 = Executor(Task + Task) + Cache 
Databricks will manage all our clusters

portal.azure.com
Resource Group = Group of services
Service = Create Azure Databricks or Create Storage Account or create anything in Azure
Data Lake = storage account
Databricks Workspace

Datalake = Outside Container -> Then a Hierarchial folder structure -> Then a list of files

Subscription = Azure Subscription 1
Microsoft Entra ID
  App Registrations > shiva_service_principle - Create AppID and TenantID
  App Registrations > shiva_service_principle > Certificates and Secrets > New Client Secret
  Then go to datalakeshiva88 > Access Control (IAM) > Add Role Assignment > Storage Blob Data Contributor > Members > search for shiva_service_principle and add > complete
  Databricks python file code config CONNECTS Service Principle CONNECTS Datalake IAM
Key vaults
   Create > Access Configuration > Permission model = "Vault access policy" > name = "masterclasskeyvaultshiva" > IAM > Assign yourself to role "Key Vault Administrator"
   masterclasskeyvaultshiva > objects > Secrets > Click Generate/Import to create a secret > Name = app-secret and provide key = secretKey
   databricksmaster_workspace URL slightly modified - added #secrets/createScope at the end - https://adb-385780289707858.18.azuredatabricks.net#secrets/createScope
     Scope Name = shivascope
     DNS Name = Key vaults > masterclasskeyvaultshiva > Settings > Properties > Vault URI
     Resource ID = Key vaults > masterclasskeyvaultshiva > Settings > Properties > Resource ID
Resource Group = RG_databricks_masterclass
  Storage Account = datalakeshiva88
    source
    destination
  Databricks = databricksmaster_workspace
    Workspace > Workspace > DatabricksMasterclass > Tutorial python file
    Compute > Create a compute cluster - One big headache since all low end servers are not available at all
Create a new Access Connector for Azure Databricks "ac-extstorage-eastus" in the same eastus region where databricks is available
  Provide/grant "Storage Blob Data Contributor" access to "ac-extstorage-eastus" on ADLS Gen2 Storage Account "datalakeshiva88" IAM. And also give access to unity-catalog access connector
    Open Databricks > catalog - External Data > Credentials  > extstorage-credential
    Open Databricks > catalog - External Data > External Locations  > extstorage-location

Magic Commands
%python
%sql
%r = to run the r commands
%md = mark down
%fs = file system
%run = to run all the contents from another python file

DBFS = Data Bricks File System
DataLake > Abstraction Layer / File Systems > URL 
/FileStore/raw/file.csv

DataBricks Python > Service Principle App Key > DataLake Data
Microsoft Entra ID > Manage > App Registrations > 
Application (client) ID : xxxx
Directory (tenant) ID   : xxxxxx
Object ID               : Object ID is not used anywhere

masterclass_client_secret
Value = value_xxxxxx
Secret ID = Secret ID is not being anywhere in the code 

-- Google search - access data lake using darabricks 
spark.conf.set("fs.azure.account.auth.type.datalakeshiva88.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.datalakeshiva88.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.datalakeshiva88.dfs.core.windows.net", "Application (client) ID")
spark.conf.set("fs.azure.account.oauth2.client.secret.datalakeshiva88.dfs.core.windows.net", "value_xxxxxx")
spark.conf.set("fs.azure.account.oauth2.client.endpoint.datalakeshiva88.dfs.core.windows.net", "https://login.microsoftonline.com/Directory (tenant) ID/oauth2/token")

Databricks CONNECTS Service Principle CONNECTS Datalake IAM

az vm list-skus --location "northeurope" --size Standard_D --all --output table
az vm list-skus --location "eastus" --output table
az vm list-skus --location "eastus" --size Standard_D --output table
az vm list-skus --location "eastus" --size Standard_D --all --output table

Standard_NC4as_T4_v3[T4]

Databricks Utilities
dbutils.fs() -- know the files available in the location

-- Delta Lake
Source data.parquet > Delta Log / Transaction Log > Destination data.parquet

External Delta Tables vs Managed Delta Tables
Managed Delta Tables
  Metastore or Hive store = Database/schema/table definition
  Cloud Storage Default = Databricks managed resource group storage 
  Deleting table in Metastore will also delete the table in Cloud Storage
External Delta Tables
  Metastore or Hive store = Database/schema/table definition
  Cloud Storage OWR OWN = Databricks managed resource group storage 
  Deleting table in Metastore will NOT delete the table in OUR OWN Cloud Storage
  










