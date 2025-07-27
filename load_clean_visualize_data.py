import numpy as np
import pandas as pd
import os
from databricks import sql
from dotenv import load_dotenv
import seaborn as sns
from matplotlib import pyplot as plt    


# loading in datafrom databricks
load_dotenv()

with sql.connect(server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME"),
                 http_path       = os.getenv("DATABRICKS_HTTP_PATH"),
                 access_token    = os.getenv("DATABRICKS_TOKEN")) as connection:

  with connection.cursor() as cursor:
    cursor.execute(f"SELECT * FROM workspace.ibm_hr.ibm_hr")
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    df = pd.DataFrame(rows, columns=columns)

#quick exploration, visualization, cleaning
# df.describe()
df.columns = df.columns.str.lower()

# filtering down to a subset of available columns
df = df[["age", 
         "attrition", 
         "department", 
         "education", 
         "distancefromhome", 
         "educationfield", 
         "environmentsatisfaction", 
         "gender", 
         "hourlyrate",
         "jobsatisfaction",
         "maritalstatus",
         "overtime",
         "performancerating",
         "relationshipsatisfaction",
         "totalworkingyears",
         "stockoptionlevel",
         "trainingtimeslastyear",
         "worklifebalance",
         "yearsatcompany",
         "yearsincurrentrole",
         "yearssincelastpromotion",
         "yearswithcurrmanager"]]

df["attrition"] = np.where(df["attrition"] == "Yes", 1, 0)
df["gender"] = np.where(df["gender"] == "Male", 1, 0)
df = pd.get_dummies(df, columns = ["maritalstatus", "department", "educationfield", "overtime"], dtype = int)

# sns.pairplot(df)

# after looking at correlation matrix, the following features seem to be correlated with eachother...
# age + totalworkingyears
# age + yearswithcompany
# yearswithcompany + totalworkingyears
# therefore dropping yearswithcompany and totalworkingyears, allowing age to be a proxy for both
df = df.drop(columns = ["totalworkingyears", "yearsatcompany"])

#checking for class imbalance
df["attrition"].value_counts(normalize = True) #lots of imbalance, so will use class_weight='balanced' in logistic regression