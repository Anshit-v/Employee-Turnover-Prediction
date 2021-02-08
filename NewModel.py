import pickle


def rforest(X_train,X_test,Y_train,Y_test):
    
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    forest= RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=0)
    forest.fit(X_train,Y_train)
    #Y_pred=forest.predict(X_test)
    pickle.dump(forest , open('rforest_model.pkl','wb'))

