import numpy as np
import pandas as pd
import os
from databricks import sql # type: ignore
from dotenv import load_dotenv # type: ignore
import seaborn as sns # type: ignore
from matplotlib import pyplot as plt  
from itertools import combinations  

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
         "performancerating",
         "relationshipsatisfaction",
         "totalworkingyears",
         "stockoptionlevel",
         "trainingtimeslastyear",
         "worklifebalance",
         "yearsincurrentrole",
         "yearssincelastpromotion",
         "yearswithcurrmanager"]]

df["attrition"] = np.where(df["attrition"] == "Yes", 1, 0)
df["gender"] = np.where(df["gender"] == "Male", 1, 0)
df = pd.get_dummies(df, columns = ["maritalstatus", "department", "educationfield"], dtype = int)

# sns.pairplot(df)

# after looking at correlation matrix, the following features seem to be correlated with eachother...
# age + totalworkingyears
# age + yearswithcompany
# yearswithcompany + totalworkingyears
# therefore dropping yearswithcompany and totalworkingyears, allowing age to be a proxy for both
df = df.drop(columns = ["totalworkingyears"])

#checking for class imbalance
df["attrition"].value_counts(normalize = True) #lots of imbalance, so will use class_weight='balanced' in logistic regression

# creating visuals for outcomes based on each column to see if there are any patterns to discern.
cols = df.drop("attrition", axis = 1)


# the problem with this is that it is not showing each point, lots of overlap and only showing the last 
# point plotted on the graph. 

'''
for col1, col2 in combinations(cols, 2):
    fg = sns.FacetGrid(data = df, hue = "attrition")
    fg.map_dataframe(sns.scatterplot, x = col1, y = col2)
    fg.add_legend()
    plt.show()
'''