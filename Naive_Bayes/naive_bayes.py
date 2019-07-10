import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data = fetch_20newsgroups()
print(data.DESCR)
for target_name in data.target_names:
    print(target_name)

# defining all categories
categories = data.target_names

# training the data on these categories
train = fetch_20newsgroups(subset='train', categories=categories)

# testing the data on these categories
test = fetch_20newsgroups(subset='test', categories=categories)

print(train.data[5])

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X=train.data, y=train.target)
labels = model.predict(test.data)

mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)


plt.xlabel("true label")
plt.ylabel("predicted label")
plt.show()


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# example
print(predict_category("audi is better than BMW"))
# print(accuracy_score())