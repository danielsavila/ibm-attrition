import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, roc_curve, recall_score, f1_score
from load_clean_visualize_data import df
import seaborn as sns
import matplotlib.pyplot as plt

X = df.drop(columns=["attrition"])
y = df["attrition"]
thresholds_list = np.linspace(0, 1, 100)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_train_pred = lda.predict_proba(X_train)


#evaluating the model via different thresholds
recall_scores = []
f1_scores = []

for threshold in thresholds_list:
    y_train_pred_thresholded = (y_train_pred[:, 1] >= threshold).astype(int)
    recall = recall_score(y_train, y_train_pred_thresholded)
    f1 = f1_score(y_train, y_train_pred_thresholded)
    recall_scores.append(recall)
    f1_scores.append(f1)

scores_df = pd.DataFrame({"recall": recall_scores, "f1": f1_scores, "threshold": thresholds_list})
optimal_threshold_f1 = scores_df.loc[scores_df['f1'].idxmax(), 'threshold']
confusion_matrix_lda = confusion_matrix(y_train, (y_train_pred[:, 1] >= optimal_threshold_f1).astype(int))
confusion_matrix_lda_display = sns.heatmap(confusion_matrix_lda, annot=True, fmt="d", cmap="Blues")
confusion_matrix_lda_display.set(xlabel="Predicted", ylabel="Label")
plt.show()
print(classification_report(y_train, (y_train_pred[:, 1] >= optimal_threshold_f1).astype(int)))


#utilizing best f1 threshold on test set
y_test_pred = lda.predict_proba(X_test)
y_test_pred_thresholded = (y_test_pred[:, 1] >= optimal_threshold_f1).astype(int)
recall = recall_score(y_test, y_test_pred_thresholded)
f1 = f1_score(y_test, y_test_pred_thresholded)
f1_score(y_test, y_test_pred_thresholded)
print(classification_report(y_test, y_test_pred_thresholded))
conf_matrix_test = confusion_matrix(y_test, y_test_pred_thresholded)
ax_test = sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues")
ax_test.set(xlabel="Predicted", ylabel="Label")
plt.show()

# Plotting ROC curve for LDA test
y_test_pred_proba = lda.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
roc_display.ax_.set_title("ROC Curve for LDA")
plt.show()

#LDA works ok, about as well as logistic regression, but performance drops off on test set.




#now trying with Quadratic Discriminant Analysis (QDA)

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_train_pred_qda = qda.predict_proba(X_train)

recall_scores = []
f1_scores = []

for threshold in thresholds_list:
    y_train_pred_qda_thresholded = (y_train_pred_qda[:, 1] >= threshold).astype(int)
    recall = recall_score(y_train, y_train_pred_qda_thresholded)
    f1 = f1_score(y_train, y_train_pred_qda_thresholded)
    recall_scores.append(recall)
    f1_scores.append(f1)

scores_df_qda = pd.DataFrame({"recall": recall_scores, "f1": f1_scores, "threshold": thresholds_list})
optimal_threshold_f1 = scores_df.loc[scores_df_qda['f1'].idxmax(), 'threshold']
confusion_matrix_qda = confusion_matrix(y_train, (y_train_pred_qda[:, 1] >= optimal_threshold_f1).astype(int))
confusion_matrix_qda_display = sns.heatmap(confusion_matrix_qda, annot=True, fmt="d", cmap="Blues")
confusion_matrix_qda_display.set(xlabel="Predicted", ylabel="Label")
plt.show()
print(classification_report(y_train, (y_train_pred_qda[:, 1] >= optimal_threshold_f1).astype(int)))

# Plotting ROC curve for QDA train
y_train_pred_proba = qda.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_proba)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
roc_display.ax_.set_title("ROC Curve for QDA")
plt.show()



#utilizing best f1 threshold on test set
y_test_pred_proba_qda = qda.predict_proba(X_test)
y_test_pred_qda_thresholded = (y_test_pred_proba_qda[:, 1] >= optimal_threshold_f1).astype(int)
print(classification_report(y_test, y_test_pred_qda_thresholded))
conf_matrix_test_qda = confusion_matrix(y_test, y_test_pred_qda_thresholded)
ax_test_qda = sns.heatmap(conf_matrix_test_qda, annot=True, fmt="d", cmap="Blues")
ax_test_qda.set(xlabel="Predicted", ylabel="Label")
plt.show()

# Plotting ROC curve for QDA test
y_test_pred_proba_qda = qda.predict_proba(X_test)[:, 1]
fpr_qda, tpr_qda, thresholds_qda = roc_curve(y_test, y_test_pred_proba_qda)
roc_display_qda = RocCurveDisplay(fpr=fpr_qda, tpr=tpr_qda).plot()
roc_display_qda.ax_.set_title("ROC Curve for QDA")
plt.show()

#QDA works better, but still underperforms on the test set. We might therefore still consider Logistic regression or KNN as better alternatives.
