# **assignment**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('accident.csv')
df.head()

df.isnull().sum()

df=df.dropna()

le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Helmet_Used']=le.fit_transform(df['Helmet_Used'])
df['Seatbelt_Used']=le.fit_transform(df['Seatbelt_Used'])

sc=StandardScaler()
df[['Age','Gender','Speed_of_Impact','Helmet_Used','Seatbelt_Used']]=sc.fit_transform(df[['Age','Gender','Speed_of_Impact','Helmet_Used','Seatbelt_Used']])

print(df)

x=df.drop('Survived',axis=1)
y=df['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_data = np.array([[0.845758, -0.893401, -1.441777, -1.119318, -1.130960]])

# Predict survival outcome
prediction = knn.predict(new_data)

print("Predicted Class:", prediction[0])


import pickle 
filename='accident.pkl'
pickle.dump(knn,open(filename,'wb'))