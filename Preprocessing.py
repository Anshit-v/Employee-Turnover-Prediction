# -*- coding: utf-8 -*-
"""
Data Preprocessing

"""
import pandas as pd


def preProcess(df):
    print("/n Checking of null values in dataset /n",df.isnull().any())
    
    print("/n Sum of null values /n ",df.isna().sum())
    
    print("/n Different values of target column ",df['left'].value_counts())
    
    #renaming column name 
    df.rename(columns={'sales': 'department'}, inplace=True)
    
    #print object data type and their unique values
    for i in df.columns:
        if df[i].dtype == object:
            print(str(i) +": "+ str(df[i].unique()))
            print(df[i].value_counts())
            print("--------")
            
    #label encoding for salary column
    # from sklearn.preprocessing import LabelEncoder
    # labelencoder = LabelEncoder()
    # df["salary"]=labelencoder.fit_transform(df["salary"])
    # print("Salary column",df["salary"].head())
    salary = {'low':0, 'medium':1,'high':2}
    df['salary'] = df['salary'].map(lambda x : salary[x])
    
    
       
    #one hot encoding for "department" column
    df1=pd.get_dummies(df)
    #columns=[department]
    
    print(df1.head())
    print(df1.dtypes)
    
    
    return df1
    
    