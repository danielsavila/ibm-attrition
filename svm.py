from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from load_clean_visualize_data import df
import numpy as np


x = df.drop("attrition", axis = 1)
y = df["attrition"]
seed = 12345
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= .3, stratify = y, random_state = seed)

skfold = StratifiedKFold(shuffle = True, random_state= seed)
svc = SVC()

# implementing stratified cross validation, and hyper paramter tuning with gridsearch

param_grid = [{"kernel": ["linear"], 
               "C": np.logspace(-3, 3, 5)},
              
              {"kernel": ["poly"],
               "C": np.logspace(-3, 3 ,5),
               "degree": [2, 3, 4, 5],
               "gamma": ["scale"],
               "coef0": np.linspace(.01, 10, 3)},
              
              {"kernel": ["rbf"],
               "C": np.logspace(-3, 3, 5),
               "gamma": ["scale"]}]

gs = GridSearchCV(estimator = svc,
                  param_grid = param_grid,
                  scoring = "f1",
                  cv = skfold,
                  n_jobs = -1)


gs.fit(x_train, y_train)
gs.best_params_
gs.best_score_

#cross_val_score(SVC(kernel = "poly", degree = 5), x_train, y_train, cv = skfold, n_jobs = -1)