# -*- coding: utf-8 -*-
"""
models
"""


from sklearn.metrics import classification_report


def model_option(X_train,X_test,Y_train,Y_test,ch):
    
    if(ch==1):
        #Logistic Regression
            from sklearn.linear_model import LogisticRegression
            #from sklearn import metrics
            lr = LogisticRegression()
            lr.fit(X_train,Y_train)
            Y_pred=lr.predict(X_test)
            print("Classification Report",classification_report(Y_test,Y_pred))
            #print(Y_pred)
            
    elif(ch==2):
        #SVM
            from sklearn.svm import SVC
            svc_classifier=SVC(kernel='linear')
            svc_classifier.fit(X_train,Y_train)
            Y_pred=svc_classifier.predict(X_test)
            print("Classification Report",classification_report(Y_test,Y_pred))
    elif(ch==3):   
        #KNN
        #by default neighbours = 5
            from sklearn.neighbors import KNeighborsClassifier
            knn_classifier=KNeighborsClassifier()
            knn_classifier.fit(X_train,Y_train)
            Y_pred=knn_classifier.predict(X_test)
            print("Classification Report",classification_report(Y_test,Y_pred))
     
    elif(ch==4):
        #ensemble technique
        #bagging
        #Random Forest
            from sklearn.ensemble import RandomForestClassifier
            forest= RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
            forest.fit(X_train,Y_train)
            Y_pred=forest.predict(X_test)
            print("Classification Report",classification_report(Y_test,Y_pred))
        
    elif(ch==5):
        #boosting
        #ADA Boost
            from sklearn.ensemble import AdaBoostClassifier
            model= AdaBoostClassifier(random_state=1)
            model.fit(X_test,Y_test)
            Y_pred=model.predict(X_test)
            print("Classification Report",classification_report(Y_test,Y_pred)) 
        
    elif(ch==6):
        #Gradient Boosting
            from sklearn.ensemble import GradientBoostingClassifier
            model=GradientBoostingClassifier(learning_rate=0.01,random_state=1)
            model.fit(X_test,Y_test)
            Y_pred=model.predict(X_test)
            print("Classification Report",classification_report(Y_test,Y_pred)) 
            
    elif(ch==7):    
        #XG boost
            import xgboost as xgb
            model=xgb.XGBClassifier(random_state=1,learning_rate=0.01,use_label_encoder=False)
            model.fit(X_test,Y_test)
            Y_pred=model.predict(X_test)
            print("Classification Report",classification_report(Y_test,Y_pred)) 
            