from load_clean_visualize_data import df
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import xgboost as xgb

x = df.drop("attrition", axis = 1)
y = df["attrition"]

seed = 12345
smote = SMOTE(random_state = seed, k_neighbors = 5)
skfold = StratifiedKFold(shuffle = True, random_state= seed, n_splits = 5)

x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = .3, 
                                                    random_state = seed, 
                                                    stratify = y)


#implement SMOTE algorithm to address class imbalances.
x_train, y_train = smote.fit_resample(x_train, y_train)


#parameters for tuning:

param_grid = {"n_estimators": np.arange(1, 300, 5),
              "max_depth": [2, 3, 4, 5],
              "min_child_weight": np.arange(7, 30, 10),
              "max_delta_step": np.arange(1, 11, 10)} #using this to attempt to further stabilize learning rates

#scaling to account for imbalanced classes
scale_weight = y_train.value_counts()[0]/y_train.value_counts()[1]

xgbc = xgb.XGBClassifier(objective = "binary:logistic",
                         grow_policy = "lossguide",
                         learning_rate = .1,
                         booster = "gbtree",
                         n_jobs = -1, 
                         random_state = seed,
                         scale_pos_weight = scale_weight
                         )

gs = GridSearchCV(estimator = xgbc, 
                  param_grid = param_grid,
                  scoring = "recall",
                  cv = skfold,
                  n_jobs = -1)

gs.fit(x_train, y_train)
gs.best_estimator_
gs.best_params_
y_pred = gs.predict(x_train)

print("training report")
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train, y_pred))

print("-"*50)

print("test report")
y_pred_test = gs.predict(x_test)
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))