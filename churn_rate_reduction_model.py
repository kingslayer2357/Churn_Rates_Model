# -*- coding: utf-8 -*-
"""
Created on Thu May 21 04:30:58 2020

@author: kingslayer
"""

# importing libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the dataset
dataset=pd.read_csv("New_churn_data.csv")

user_identifier=dataset["userid"]
dataset=dataset.drop(columns=["userid"])

#One Hot Encoding
dataset=pd.get_dummies(dataset)
dataset.columns
dataset=dataset.drop(columns=["payfreq_na","zodiac_sign_na","rent_or_own_na"])


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset.drop(columns=["churn"]),dataset["churn"],test_size=0.2,random_state=0)


#Baliancing
pos_index=y_train[y_train.values==1].index
neg_index=y_train[y_train.values==0].index

if len(pos_index)>len(neg_index):
    higher=pos_index
    lower=neg_index
else:
    lower=pos_index
    higher=neg_index
    
random.seed(0)
higher=np.random.choice(higher,size=len(lower))
lower=np.asarray(lower)
new_indexes=np.concatenate((higher,lower))

X_train=X_train.loc[new_indexes,]
y_train=y_train[new_indexes]

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train2=pd.DataFrame(sc_X.fit_transform(X_train))
X_test2=pd.DataFrame(sc_X.transform(X_test))
X_train2.columns=X_train.columns.values
X_train2.index=X_train.index.values
X_test2.columns=X_test.columns.values
X_test2.index=X_test.index.values
X_train=X_train2
X_test=X_test2


#Model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)


y_pred=classifier.predict(X_test)



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,scoring="accuracy",cv=10)
mean_accuracy=accuracies.mean()
deviation=accuracies.std()
print(f"Accuracy: {mean_accuracy} +- {deviation}")


#feature Selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
rfe=RFE(classifier,20)
rfe=rfe.fit(X_train,y_train)



classifier.fit(X_train[X_train.columns[rfe.support_]],y_train)


y_pred=classifier.predict(X_test[X_test.columns[rfe.support_]])



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,scoring="accuracy",cv=10)
mean_accuracy=accuracies.mean()
deviation=accuracies.std()
print(f"Accuracy: {mean_accuracy} +- {deviation}")

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)

final_result=pd.concat([y_test,user_identifier],axis=1).dropna()
final_result["prediction"]=y_pred
final_result[["userid",'churn',"prediction"]].reset_index(drop=True)