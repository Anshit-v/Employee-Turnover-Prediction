# -*- coding: utf-8 -*-
""" main """

import pandas as pd
from DataInfo import *
from Models import *
from Preprocessing import *
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from NewModel import rforest

df=pd.read_csv(r'C:/Users/Anshit/Downloads/datasets_9768_13874_HR_comma_sep.csv')

#df.info()
dataInfo(df)
df1=preProcess(df)

Y=df1["left"]
X=df1.loc[:, df1.columns != 'left']

#feature selection

# bestfeatures = SelectKBest(score_func=chi2, k=10)
# fit = bestfeatures.fit(X,Y)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X.columns)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(10,'Score'))  #print 10 best features

#split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)


# SMOTE method for upsampling so that data will be balanced

smt=SMOTE()
X_train_sm, Y_train_sm= smt.fit_resample(X_train,Y_train)


rforest(X_train_sm,X_test,Y_train_sm,Y_test)


'''
flag=True
while(flag==True):
    print(" \n 1. Logistic Regression \n \
2. Support vector machine \n \
3. K Neighbours Classifier \n 4. Random Forest Classifier \n \
5. Gradient Boosting Classifier \n 6. Ada Boost \n 7. XG Boost \n \
8. exit")
    ch=int(input("Enter the model no: "))
 
    
    if 0<ch<8:
        model_option(X_train_sm,X_test,Y_train_sm,Y_test,ch)
    elif ch==8:
        flag=False
    else:
        print("enter proper input")
        
print("thank you")
'''