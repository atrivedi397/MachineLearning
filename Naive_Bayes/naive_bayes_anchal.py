# Import LabelEncoder
import pandas as pd

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# creating labelEncoder for normalising the label
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
df = pd.read_csv("/run/media/atrivedi/Drive 1 (NTFS)/Machine-Learning/DataSets For ML/weather.csv")

"""Encoding is done in an alphabetical order"""
weather_encoded = le.fit_transform(df.weather)
print("WEATHER ENCODE : ", weather_encoded)

temprature_encoded = le.fit_transform(df.temp)
print("TEMPERATURE ENCODE : ", temprature_encoded)

play_encoded = le.fit_transform(df.play)
print("PLAY ENCODE : ", play_encoded)

# Combining weather and temp into single list of tuples
features = zip(weather_encoded, temprature_encoded)
features = list(features)
print(list(features))

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features, play_encoded)

# weather range = [0: overcast, 1: rainy, 2: sunny]
# temp range = [0: cool, 1: hot, 2: mild]
# Predict Output
predicted = []
for weather, temp in zip(weather_encoded, play_encoded):
    predicted.append(model.predict([[weather, temp]]))    # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)
print(confusion_matrix(y_true=play_encoded, y_pred=predicted))
print(classification_report(y_pred=predicted, y_true=play_encoded))
print("Accuracy of the model : ", accuracy_score(y_pred=predicted, y_true=play_encoded))
