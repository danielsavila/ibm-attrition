from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from load_clean_visualize_data import df
import numpy as np


x = df.drop("attrition", axis = 1)
y = df["attrition"]
seed = 12345


#scaling is necessary in SVM since large magnitude models will influence distance calculations.
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= .3, stratify = y, random_state = seed)

skfold = StratifiedKFold(shuffle = True, random_state= seed, n_splits = 3)
svc = SVC(class_weight = "balanced")

# implementing stratified cross validation, and hyper paramter tuning with gridsearch

param_grid = [{"kernel": ["linear"], 
               "C": np.logspace(-3, 3, 5)},
              
              {"kernel": ["poly"],
               "C": np.logspace(-3, 3 ,5),
               "degree": [2, 3, 4],
               "gamma": ["auto"],
               "coef0": np.linspace(.01, 10, 3)},
              
              {"kernel": ["rbf"],
               "C": np.logspace(-3, 3, 5),
               "gamma": ["auto"]}]


param_grid2 = {"kernel":["poly"],
               "C": np.logspace(-3, 3, 15),
               "gamma": ["auto"],
               "coef0": np.linspace(.01, 10, 5)}

gs = GridSearchCV(estimator = svc,
                  param_grid = param_grid,
                  scoring = "f1",
                  cv = skfold,
                  n_jobs = -1)


gs.fit(x_train, y_train)
gs.best_params_
gs.best_score_

best_model = gs.best_estimator_
y_pred = best_model.predict(x_train)
y_pred_test = best_model.predict(x_test)

print("training report")
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train,y_pred))
print("-"*50)

print("test report")
print(confusion_matrix(y_test, y_pred_test))
print(classification_report(y_test, y_pred_test))

# SVM is having trouble with the class imbalances. 
# In every mdoel that I train, the problem seems to be that we are leaning towards predicting the majority class
