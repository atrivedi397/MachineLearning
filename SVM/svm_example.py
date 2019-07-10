# imported for some built in data sets
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# for the training of model
from sklearn.model_selection import train_test_split

# for SVM
from sklearn import svm

# for testing of the predicted output
from sklearn import metrics

# for plotting features
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

petals_panda = pd.read_csv("/run/media/atrivedi/Drive 1 (NTFS)/Machine-Learning/DataSets For ML/IRIS.csv")

# built in data sets
petals = datasets.load_iris()
print(petals.DESCR)
print("Features : ", petals.feature_names)
print("Target classes : ", petals.target_names)
print(petals.data)

print(petals.data.shape)
print(petals.target)
print(len(petals.target))
x = petals_panda.iloc[:, :-1]
y = petals_panda.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

# for model evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# for plotting the details
sns.pairplot(data=petals_panda, hue='species', palette='Set1')
plt.show()

print("Accuracy of the model : ", accuracy_score(y_pred=y_pred, y_true=y_test))
