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
## X Data:
![image](https://github.com/user-attachments/assets/c0d06c53-17fc-4da1-9c94-d87c24cb1c2e)


## Y Data:
![image](https://github.com/user-attachments/assets/4ac3846b-88b5-475b-826a-72d173024d6a)


## Predicted Values:
![image](https://github.com/user-attachments/assets/df8e2395-cc22-42ba-b53a-7be59d566f2a)

## Accuracy Score:
![image](https://github.com/user-attachments/assets/ca869d50-308b-4469-a868-27be3ac6d323)


## Confusion Matrix:
![image](https://github.com/user-attachments/assets/0ffb5e86-a88b-4042-bea8-4a914e27e669)


## Classification Report:
![image](https://github.com/user-attachments/assets/7f52a854-ca18-4d7b-aa2d-7139906a3384)


## Predicting output from Regression Model
![image](https://github.com/user-attachments/assets/8aa346b8-aa37-41df-ad13-3063ab79b425)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
