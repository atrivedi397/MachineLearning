import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=40, centers=2, random_state=20)

clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)


# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.plot(range(18))
plt.show()

newData = [[3, 4], [5, 6]]
print(clf.predict(newData))
"""
df = pd.DataFrame({'x_dimension': [9, 15, 25, 14, 10, 18, 0, 16, 5, 19, 16, 20],
                   "y_dimension": [39, 56, 93, 61, 50, 75, 32, 85, 42, 70, 66, 80]})

new_grp = df.groupby('x_dimension')
print(new_grp.get_group(20))
"""
