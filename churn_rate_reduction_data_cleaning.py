# -*- coding: utf-8 -*-
"""
Created on Thu May 21 03:32:06 2020

@author: kingslayer
"""

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the dataset

dataset=pd.read_csv("Churn_data.csv")
dataset.describe()

##EDA

#Cleaning Data

#Removing NaN

dataset.isna().any()
dataset.isna().sum()
dataset=dataset[pd.notnull(dataset["age"])]
dataset=dataset.drop(columns=["credit_score","rewards_earned"])

#Histograms

dataset2=dataset.drop(columns=["user","churn"])
plt.suptitle("Histograms of features",fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.figure(figsize=(50,50))
    plt.subplot(6,5,i)
    f=plt.gca()
    f.set_title(dataset2.columns.values[i-1])
    
    vals=np.size(dataset2.iloc[:,i-1].unique())
    
    plt.hist(dataset2.iloc[:,i-1],bins=vals,color="green")
    plt.show()
    
    
#Pie Charts
plt.suptitle("Pie Charts of features",fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.figure(figsize=(50,50))
    plt.subplot(6,5,i)
    f=plt.gca()
    f.set_title(dataset2.columns.values[i-1])
    
    vals=dataset2.iloc[:,i-1].value_counts(normalize=True).values
    index=dataset2.iloc[:,i-1].value_counts(normalize=True).index
    plt.axis('equal')
    plt.pie(x=vals,labels=index,autopct='%1.1f%%')
    plt.show()
    
    
# Correlation Plot


dataset.drop(columns=["user","churn","housing","payment_type","zodiac_sign"]).corrwith(dataset.churn).plot.bar(figsize=(20,10),title="Correlation",grid=True,rot=45)


#Correlation Matrix
plt.figure(figsize=(20,20))
corr=dataset.corr()
sns.heatmap(corr,annot=True)


dataset=dataset.drop(columns=["app_web_user"])
dataset.to_csv("new_data.csv")
