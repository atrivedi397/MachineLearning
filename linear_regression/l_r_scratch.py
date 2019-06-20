"""
Is there a relationship between the daily minimum and maximum temperature? 
Can you predict the maximum temperature given the minimum temperature?
"""""

import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# place your path to the csv file here
data = pd.read_csv("/run/media/atrivedi/Drive 1 (NTFS)/Machine-Learning/DataSets For ML/Summary of Weather.csv")
print(data.shape)
# print(data.head(25))

# Step 1 -  Get the X-axis and Y-axis values or dependent and independent values
X = data["MinTemp"].values
Y = data["MaxTemp"].values

# Step 2 - Calculate the mean of the these variables
mean_X = sum(X)/len(X)
mean_Y = sum(Y)/len(Y)

# Step 3 - Fit a line y = m*x +c  where y = mean_Y, x = mean_X, m = sum ( (x-mean_x) (y-mean_y) ) / (sum (x-mean_x))^2
numerator, denominator = 0, 0

# calculating 'm' and 'c'
for i in range(len(X)):
    numerator += (X[i]-mean_X) * (Y[i]-mean_Y)
    denominator += (X[i]-mean_X)**2


slope = numerator/denominator
intercept = mean_Y - (slope * mean_X)

# printing 'm' and 'c'
print(f"slope= {slope} and intercept = {intercept}")
print(f"The line equation becomes y = {slope}*x + {intercept}")

# Step 4 - Plotting a regression line

# scaling the scattered data to small 2D plot
max_x, max_y = np.max(X)+100, np.max(Y)-100

# getting multiple points on x(independent variable) to get the corresponding values of y(dependent variable)
x = np.linspace(max_x, max_y)

# calculating the corresponding values for y
y = slope*x + intercept
# print(y)
plt.scatter(X, Y, c='#FA8072', label='Data Given')
plt.plot(x, y, c='b', label='Regression Line')
plt.xlabel("Min Temp")
plt.ylabel("Max Temp")
plt.legend()
plt.show()

"""Using R-Squared method to check how close is our model to actual data"""

# Step 1 - Now predict some values by the now traced line y = mx+c
"""
R-squared = (sum(Y-predicted - Y_Mean))**2/ (sum(Actual Y - Y_Mean))**2
"""

rsq_numerator, rsq_denominator = 0, 0
y_pred_array = []
for i in range(len(X)):
    y_pred = slope*X[i] + intercept
    y_pred_array.append(y_pred)
    rsq_numerator += (y_pred-mean_Y)**2
    rsq_denominator += (Y[i]-mean_Y)**2

print(f"R-Squared value of given model is {rsq_numerator/rsq_denominator}")

# calculating how much %age of predicted model matches with the actual given data
"""
% of correctness =        mean of predicted values
                    ---------------------------------------- * 100
                    mean of  dependent variable given values  
"""

percentage_of_accuracy = ((sum(y_pred_array)/len(y_pred_array)) / mean_Y) * 100

"""It is solely based on the mean of the predicted values and actual values hence , 
it will tell you that the model is 100% correct when mean is same of both values but,
the difference in the actual value and predicted value will more precisely tell the accuracy of the model"""

print(f"The correctness of the model is {percentage_of_accuracy}%")


if __name__ == "__main__":
    dates = data["Date"]
    print(type(dates))
    date_list = list(dates)
    print("This model predicts the maximum temperature of the day for a given minimum temperature")
    start_date, end_date = "1942-7-1", "1945-12-31"
    input_date = input(f"Provide the date between {start_date, end_date} for which you want to check the "
                       "predicted value and actual value in format (yyyy-mm-dd)")

    if input_date in date_list:
        actual_max = dates[input_date].get("MaxTemp")
        actual_min = dates[input_date].get("MinTemp")
        print(f"The value of the Minimum temp is {actual_min}")
        print(f"The actual value of the max temp is {actual_max}")
        print(f"The predicted value of the max temp is {slope*actual_min + intercept}")
    else:
        print("You have entered an invalid or out of range date")
