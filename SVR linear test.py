
# coding: utf-8

# In[8]:


import numpy as np
import time
from sklearn import metrics, svm
import pandas as pd
from sklearn import preprocessing


# In[9]:


from sklearn.cross_validation import train_test_split
data = pd.read_csv('randomforest_selectedfeatures.csv')
labels = data.columns.tolist()


X = data.drop(['PER', 'Player', 'player_efficiency'], 1).fillna(0) #'player_efficiency', 'Unnamed: 0', 'Team'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[10]:


X = scaler.fit_transform(X)
y = data.PER

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


# Calculate everything

import time
from sklearn import metrics

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    start = time.clock()
    lab_enc = preprocessing.LabelEncoder()
    y_test2 = lab_enc.fit_transform(y_test)
    model.fit(X_train, y_train)
    kfold = 10
    
    print ("Accuracy on training set for", name, ":")
    print (model.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (model.score(X_test, y_test))
    
    y_pred = model.predict(X_test)
    y_pred2 = lab_enc.fit_transform(y_pred)
    
    print ("MAE score:")
    print (metrics.mean_absolute_error(y_test, y_pred))
    scoring = 'neg_mean_absolute_error'
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("MAE: ", results.mean(), results.std())
    
    print ("MSE score:")
    print (metrics.mean_squared_error(y_test, y_pred))
    scoring = 'neg_mean_squared_error'
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("MSE: ", results.mean(), results.std())
    
    print ("R^2 score:")
    print (metrics.r2_score(y_test, y_pred))
    scoring = 'r2'
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("R^2: ", results.mean(), results.std())
    end = time.clock()
    print('That took', end-start, 'seconds')
#     print ("Confusion Matrix:")
#     print (metrics.confusion_matrix(y_test, y_pred))
    print('\n')


# In[ ]:


kern = ['linear']
epsil = [0.1, 0.2, 0.5, 0.001, 0.0001, 0.000001, 0.0000001]
c_value = [1.0, 10, 100, 1000, 10000]

for knl in kern:
    for c in c_value:
        for eps in epsil:
            train_and_evaluate('SVR'+knl+'_'+str(c)+'-'+str(eps), svm.SVR(kernel=knl, C=c, epsilon=eps), X_train, X_test, y_train, y_test)
            print('\n')

