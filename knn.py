from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from load_clean_visualize_data import df
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


X = df.drop(columns=["attrition"])
y = df["attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)


kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=12345)

k_range = range(1, 21)
mean_accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_train_pred = knn.predict(X_train)
    accuracy = cross_val_score(knn, X_train, y_train, cv=kf, scoring='f1_weighted')
    mean_accuracies.append(accuracy.mean())

best_k = k_range[np.argmax(mean_accuracies)]
best_score = max(mean_accuracies)

best_k
best_score

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
print(classification_report(y_train, y_train_pred))
conf_matrix = confusion_matrix(y_train, y_train_pred)
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
ax.set(xlabel="Predicted", ylabel="Label")
plt.show()

y_test_pred = knn.predict(X_test)
print(classification_report(y_test, y_test_pred))
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
ax_test = sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues")
ax_test.set(xlabel="Predicted", ylabel="Label")
plt.show()