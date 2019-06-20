from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

petals = datasets.load_iris()
print("Features : ", petals.feature_names)
print("Target classes : ", petals.target_names)
print(petals.data.shape)
print(petals.target)

x_train, x_test, y_train, y_test = train_test_split(petals.data, petals.target, test_size=0.2)

clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print("Precision :", metrics.precision_score(y_test, y_pred))
print("Recall: ", metrics.recall_score(y_test, y_pred))
