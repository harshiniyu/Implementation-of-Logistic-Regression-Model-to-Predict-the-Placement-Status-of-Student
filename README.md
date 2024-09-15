# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
2.Import LabelEncoder and encode the dataset.
3.Import LogisticRegression from sklearn and apply the model on the dataset. 
4.Predict the values of array
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 
6.Apply new unknown values 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Harshini Y
RegisterNumber:  212223240050
*/

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2024-09-12 140424](https://github.com/user-attachments/assets/2c4c669c-52e1-46f6-a014-e0b0cd53e53e)

## Original data(first five columns):
![Screenshot 2024-09-12 140441](https://github.com/user-attachments/assets/0dfd3082-967f-4071-a04e-958b27660923)

## Data after dropping unwanted columns(first five):
![Screenshot 2024-09-12 140451](https://github.com/user-attachments/assets/58a8dc14-467d-4ca1-a956-7c3b85da7450)


## Checking the presence of null values:
![Screenshot 2024-09-15 165927](https://github.com/user-attachments/assets/3374d39d-4de2-4acb-aa8c-1fccfa77a626)

## Checking the presence of duplicated values:
![Screenshot 2024-09-15 171349](https://github.com/user-attachments/assets/a22c14a6-e99e-4582-8cfe-bc159892167c)

## Data after Encoding:
![Screenshot 2024-09-15 171426](https://github.com/user-attachments/assets/c22f32a6-893d-431a-9a6f-953a392665ab)
## X Data:
![Screenshot 2024-09-15 171510](https://github.com/user-attachments/assets/79922d1b-15b4-452a-ba91-3b11ad103ae4)

## Y Data:
![Screenshot 2024-09-15 171553](https://github.com/user-attachments/assets/146c0629-6e22-43a3-8729-2223c752f8c6)

## Predicted Values:
![Screenshot 2024-09-15 171646](https://github.com/user-attachments/assets/5105faa9-1772-494b-b37b-1c9886a9e8cd)

## Accuracy Score:
![Screenshot 2024-09-15 171831](https://github.com/user-attachments/assets/1b37c6fd-371d-44c4-81ab-57f8903c22e5)

## Confusion Matrix:
![Screenshot 2024-09-15 171838](https://github.com/user-attachments/assets/d6e7f770-f812-4c1f-9ee3-63a82317343b)

## Classification Report:
![Screenshot 2024-09-15 171914](https://github.com/user-attachments/assets/e8b2cdd5-763a-416c-8f1a-1e123493ea00)

## Predicting output from Regression Model
![Screenshot 2024-09-15 172131](https://github.com/user-attachments/assets/be85a07d-052d-42ae-b79b-655cb2a414ea)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
