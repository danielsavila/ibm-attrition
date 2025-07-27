from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay
from load_clean_visualize_data import df
import seaborn as sns
import numpy as np
import pandas as pd


df.head()

X = df.drop(columns=["attrition"])
y = df["attrition"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state= 12345)
log_reg = LogisticRegression(max_iter=1000, random_state=12345, class_weight='balanced')
log_reg.fit(X_train, y_train)
y_train_pred = log_reg.predict(X_train)
y_train_pred_proba = log_reg.predict_proba(X_train)[:, 1]

# looking at intial confusion matrix and classification report
conf_matrix = confusion_matrix(y_train, y_train_pred)
ax = sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues")
ax.set(xlabel = "Predicted", ylabel = "Label")
print(classification_report(y_test, y_train_pred))
accuracy_score(y_test, y_train_pred)

'''
tuning this model depends on what we are trying to optimize.
balancing the true and false positive rates would lead us to optimize f1/accuracy,
but if the cost of targeting someone who is going to leave is low, than we might be more tolernat of false positives.
alternatively, if the cost of not targeting someone who is going to leave is low, then we might be more tolerant of false negatives,
both of which would have an impact on the "optimal" threshold for this model.
'''

# plotting roc curve to visually determine optimal threshold, with a preference towards balancing false positives and true positives
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_proba)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
roc_display.ax_.set_title("ROC Curve")

# finding the optimal threshold
thresholds_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
thresholds_df[(thresholds_df["fpr"] < .4) & (thresholds_df["tpr"] < .9)].sort_values("tpr", ascending=False).head(5) #seems to be around .39

#checking accuracy with new threshold
optimal_threshold = .39
y_train_pred_optimal = np.where(y_train_pred_proba >= optimal_threshold, 1, 0)
conf_matrix_optimal = confusion_matrix(y_train, y_train_pred_optimal)
ax_optimal = sns.heatmap(conf_matrix_optimal, annot=True, fmt="d", cmap="Blues")
ax_optimal.set(xlabel="Predicted", ylabel="Label")
print(classification_report(y_train, y_train_pred_optimal))
