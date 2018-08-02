
# coding: utf-8

# In[30]:


'''
Phase 1: Feature extraction using PCA with SVR
1. Use PCA to reduce the dimensionality of the data.
2. Pick the principal components that will generate the lowest MSE error.
3. Use the reduced dimensional features as inputs for the SVR model.
4. Calculate R square score, MAE, MSE, RMSE for PCA-SVR. 

The general steps of PCA are shown as follow:
1. Normalize data
2. Calculate covariance matrix and get the eigenvalues and eigenvectors
3. Choosing principal components and forming feature vector
4. Transform original dataset to get the k-dimensional feature subspace 

'''


# In[153]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

from sklearn.cross_validation import train_test_split
data = pd.read_csv('players5years.csv')
labels = data.columns.tolist()
X = data.drop('per', 1).drop('player_efficiency', 1).drop('player_id', 1).fillna(0)
X[X<0] = 0
y = data.per

X_train, X_test, y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

# In[180]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[181]:


X_scaled


# In[55]:


from sklearn.decomposition import PCA
X_train = X.iloc[X_train]
X_test = X.iloc[X_test]

n_col = X_train.shape[1]
pca = PCA(n_components=17)

train_components = pca.fit_transform(X_train)
test_components = pca.transform(X_test)

print("original shape:   ", X_train.shape)
print("transformed shape:", train_components.shape)


# Dump components relations with features:
pd.DataFrame(pca.components_,columns=X.columns,index = ['PC-1','PC-2', 'PC-3','PC-4', 'PC-5','PC-6', 'PC-7','PC-8', 
                                                                        'PC-9','PC-10', 'PC-11','PC-12', 'PC-13','PC-14', 'PC-15','PC-6', 'PC-17'])


# In[24]:


pca.components_


# In[25]:


pca.get_covariance()


# In[26]:


explained_variance = pca.explained_variance_ratio_
explained_variance


# In[27]:


plt.plot(np.cumsum(explained_variance))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
'''
'''
The image shows that we need a max of 17 features
'''
'''

# In[28]:


# plt.scatter(train_components[:, 0], train_components[:, 1],
#             c=y_train, edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('spectral', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar();


# In[56]:


# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))

#     plt.bar(range(4), explained_variance, alpha=0.5, align='center',
#             label='individual explained variance')
#     plt.step(range(4), cum_var_exp, where='mid',
#              label='cumulative explained variance')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal components')
#     plt.legend(loc='best')
#     plt.tight_layout()


# In[58]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train2 = scaler.transform(X_train)
X_test2 = scaler.transform(X_test)


# In[59]:


from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.95)


# In[60]:


pca.fit(X_train)


# In[61]:


X_train2 = pca.transform(X_train2)
X_test2 = pca.transform(X_test2)


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')


# In[108]:


#We need to convert the labels otherwise it will give LogisticRegression: Unknown label type: 'continuous' using sklearn in python

from sklearn import preprocessing

lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)

logisticRegr.fit(X_train2, y_train)


# In[97]:


from sklearn.utils import check_consistent_length


# In[105]:


check_consistent_length(X_train2, y_train)


# In[102]:


X_train2


# In[107]:


y_train


# In[ ]:
'''

import numpy as np
import time
from sklearn                        import metrics, svm
from sklearn.linear_model           import LinearRegression #GLM
from sklearn.linear_model           import LogisticRegression #GLM
from sklearn.linear_model           import BayesianRidge #GLM
from sklearn.tree                   import DecisionTreeRegressor
from sklearn.neighbors              import KNeighborsRegressor
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.naive_bayes            import GaussianNB
from sklearn.svm                    import SVC
from sklearn.metrics                import classification_report
from sklearn.linear_model           import Ridge
from sklearn.tree                   import DecisionTreeRegressor
from sklearn.dummy                  import DummyRegressor
from sklearn                        import preprocessing
trainingData    = X_train #np.array([ [2.3, 4.3, 2.5],  [1.3, 5.2, 5.2],  [3.3, 2.9, 0.8],  [3.1, 4.3, 4.0]  ])
trainingScores  = y_train#np.array( [3.4, 7.5, 4.5, 1.6] )
predictionData  = X_test#np.array([ [2.5, 2.4, 2.7],  [2.7, 3.2, 1.2], [2.6, 2.4, 2.5], [2.5, 2.4, 2.7],  [2.7, 3.2, 1.2], [2.6, 2.4, 2.5]  ])



lab_enc = preprocessing.LabelEncoder()
trainingScores = lab_enc.fit_transform(trainingScores)
y_test2 = lab_enc.fit_transform(y_test)


start = time.clock()
clf = LinearRegression()
clf.fit(trainingData, trainingScores)
print("LinearRegression")
y_pred = clf.predict(predictionData)
# print(y_pred)
print("Score LinearRegression: ", clf.score(X_test, y_test2))#, clf.coef_)
end = time.clock()
print('Total time: ', end-start, 'seconds')
print('\n')


start = time.clock()
kern = ['rbf', 'sigmoid']#, 'poly', 'linear']
epsil = [0.1, 0.2, 0.5, 0.001, 0.0001, 0.000001, 0.0000001]
for knl in kern:
    for eps in epsil:
        clf = svm.SVR(kernel=knl, C=1.0, epsilon=eps)
        clf.fit(trainingData, trainingScores)
        print("SVR with kernel ", knl, "and epsilon ", eps)
        y_pred = clf.predict(predictionData)
        # print(y_pred)
        print("Score SVR: ", clf.score(X_test, y_test2))
        end = time.clock()
        print('Total time: ', end-start, 'seconds')
        print('\n')

        start = time.clock()
clf = LogisticRegression()
clf.fit(trainingData, trainingScores)
print("LogisticRegression")
y_pred = clf.predict(predictionData)
# print(y_pred)
print("Score LogisticRegression: ", clf.score(X_test, y_test2))
end = time.clock()
print('Total time: ', end-start, 'seconds')
print('\n')

start = time.clock()
clf = BayesianRidge()
clf.fit(trainingData, trainingScores)
print("BayesianRidge")
y_pred = clf.predict(predictionData)
# print(y_pred)
print("Score BayesianRidge: ", clf.score(X_test, y_test2))
end = time.clock()
print('Total time: ', end-start, 'seconds')
print('\n')

start = time.clock()
clf = DecisionTreeRegressor()
clf.fit(trainingData, trainingScores)
print("DecisionTreeRegressor")
y_pred = clf.predict(predictionData)
# print(y_pred)
print("Score DecisionTreeRegressor: ", clf.score(X_test, y_test2))
end = time.clock()
print('Total time: ', end-start, 'seconds')
print('\n')

start = time.clock()
clf = KNeighborsRegressor(n_neighbors=2)
clf.fit(trainingData, trainingScores)
print("KNeighborsRegressor")
y_pred = clf.predict(predictionData)
# print(y_pred)
print("Score KNeighborsRegressor: ", clf.score(X_test, y_test2))

end = time.clock()
print('Total time: ', end-start, 'seconds')
print('\n')

# ridge_regression
start = time.clock()
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

for alpha in alpha_ridge:
    clf = Ridge(alpha=alpha,normalize=True)
    clf.fit(trainingData, trainingScores)
    print("Ridge regression alpha: " + str(alpha))
    y_pred = clf.predict(predictionData)
    print("Score Ridge regression: ", clf.score(X_test, y_test))
print('Total time: ', end-start, 'seconds')
print('\n')   
    


# In[ ]:


# from sklearn import datasets
# from sklearn.model_selection import cross_val_predict
# from sklearn import linear_model
# import matplotlib.pyplot as plt

# lr = linear_model.LinearRegression()
# boston = datasets.load_boston()
# y = boston.target

# # cross_val_predict returns an array of the same size as `y` where each entry
# # is a prediction obtained by cross validation:
# predicted = cross_val_predict(lr, boston.data, y, cv=10)

# fig, ax = plt.subplots()
# ax.scatter(y, predicted, edgecolors=(0, 0, 0))
# ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()


# # Mean Absolute Error

# In[ ]:


# Cross Validation Regression MAE
'''
The Mean Absolute Error (or MAE) is the sum of the absolute differences between predictions and actual values. 
It gives an idea of how wrong the predictions were.

The measure gives an idea of the magnitude of the error, but no idea of the direction (e.g. over or under predicting).
A value of 0 indicates no error or perfect predictions. Like logloss, this metric is inverted by the cross_val_score() function.''''''


# In[197]:


from sklearn import model_selection
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
results = model_selection.cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
print("MAE: ", results.mean(), results.std())

from sklearn.metrics import mean_absolute_error
print("MAE sklearn", mean_absolute_error(y_test2, y_pred))


# # Mean Squared Error

# In[ ]:


# Cross Validation Regression MSE
'''
'''The Mean Squared Error (or MSE) is much like the mean absolute error in that it provides a gross idea of the 
magnitude of error.
Taking the square root of the mean squared error converts the units back to the original units of the output 
variable and can be meaningful for description and presentation. 
This is called the Root Mean Squared Error (or RMSE).
This metric too is inverted so that the results are increasing. Remember to take the absolute value before taking the square root if you are interested in calculating the RMSE.''''''


# In[199]:


from sklearn import model_selection
model = LinearRegression()
scoring = 'neg_mean_squared_error'
results = model_selection.cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
print("MSE: ", results.mean(), results.std())

from sklearn.metrics import mean_squared_error
print("MSE sklearn", mean_squared_error(y_test, y_pred))


# # R^2 Metric

# In[ ]:


# Cross Validation Regression R^2

The R^2 (or R Squared) metric provides an indication of the goodness of fit of a set of predictions to the actual values. In statistical literature, this measure is called the coefficient of determination.

This is a value between 0 and 1 for no-fit and perfect fit respectively.
You can see that the predictions have a poor fit to the actual values with a value close to zero and less than 0.5.
'''


# In[206]:

'''
from sklearn import model_selection
model = LinearRegression()
scoring = 'r2'
results = model_selection.cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
print("R^2: ", results.mean(), results.std())

from sklearn.metrics import r2_score
print("R^2 sklearn", r2_score(y_test2, y_pred))

'''
# In[267]:


# Calculate everything

from sklearn import model_selection
import time
from sklearn import metrics

def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    start = time.clock()
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
    print (metrics.mean_absolute_error(y_test2, y_pred))
    scoring = 'neg_mean_absolute_error'
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("MAE: ", results.mean(), results.std())
    
    print ("MSE score:")
    print (metrics.mean_squared_error(y_test, y_pred))
    scoring = 'neg_mean_squared_error'
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("MSE: ", results.mean(), results.std())
    
    print ("R^2 score:")
    print (metrics.r2_score(y_test2, y_pred))
    scoring = 'r2'
    results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    print("R^2: ", results.mean(), results.std())
    end = time.clock()
    print('That took', end-start, 'seconds')
#     print ("Confusion Matrix:")
#     print (metrics.confusion_matrix(y_test, y_pred))
    print('\n')


# In[306]:


train_and_evaluate('LinearRegression', LinearRegression(), X_train, X_test, y_train, y_test)
print('\n')

train_and_evaluate('SVR', svm.SVR(), X_train, X_test, y_train, y_test)
print('\n')

train_and_evaluate('BayesianRidge', BayesianRidge(), X_train, X_test, y_train, y_test)
print('\n')

depth = [3, 5, 10, 50, 100]
for num in depth:
    train_and_evaluate('DecisionTreeRegressor depth: '+str(num), DecisionTreeRegressor(max_depth = num, random_state=0), X_train, X_test, y_train, y_test)
print('\n')

neighbours = [5, 10, 50, 100]
for neighbour in neighbours:
    train_and_evaluate('KNeighborsRegressor', KNeighborsRegressor(n_neighbors=neighbour), X_train, X_test, lab_enc.fit_transform(y_train), lab_enc.fit_transform(y_test))
train_and_evaluate('Standard Ridge', Ridge(alpha=alpha,normalize=True), X_train, X_test, y_train, y_test)
train_and_evaluate('DummyRegressor', DummyRegressor(strategy='mean', constant=None, quantile=None), X_train, X_test, y_train, y_test)


# In[ ]:
'''

#Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_simple


# In[302]:


from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors].reshape(-1, 1),data['per'])
    y_pred = ridgereg.predict(data[predictors].reshape(-1, 1))

#     #Check if a plot is to be made for the entered alpha
#     if alpha in models_to_plot:
#         plt.subplot(models_to_plot[alpha])
#         plt.tight_layout()
#         plt.plot(data['x'],y_pred)
#         plt.plot(data['x'],data['y'],'.')
#         plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['per'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret


# In[303]:


#Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    print(i)
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, 'per', alpha_ridge[i], models_to_plot)


# In[290]:


y


# In[280]:


alpha_ridge[0]
'''
