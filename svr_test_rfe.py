
# coding: utf-8

# In[1]:


# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas as pd
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# load data
dataframe = pd.read_csv('created_data/players5years.csv')
X = dataframe.drop(['player_id', 'player_efficiency', 'per'], 1).fillna(0).values
y = dataframe.per.values
names = dataframe.drop(['player_id', 'player_efficiency', 'per'], 1).columns.tolist()
dataframe.head(3)


# In[2]:


# Rescale data (between 0 and 1), rescale range to (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)


# In[3]:


# Standardize data (0 mean, 1 stdev), standardize data so that variance is 1 and mean is zero.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
standardizedX = scaler.transform(X)


# In[4]:


"""runModel is a generic function that runs a given model and produces the r2 score with or without cross-validation.
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def runModel(model, Xw, y, cv):
    if cv==False:
        model.fit(Xw,y)
        score = model.score(Xw,y)     
    else:
        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        scores = cross_val_score(model, Xw, y, cv=kfold, scoring='r2')
        score = np.array(scores).mean()
    return(score)


# In[5]:


"""RFAperf runs the given model using the provide feature ranking list starting with the top feature and adding 
the next important feature in a recursive fashion and reports the score at each step along with the features 
utilized in that step. RFA stands for Recursive Feature Augmentation."""
def RFAperf(ranking, modelstr, Xw, y, names, cv=False):
    ranking = list(ranking)
    model = eval(modelstr)
    l = len(ranking)
    
    f_inds = []
    f_names = np.array([])
    f_scorelist = []
    for i in range(1,l+1):
        f_ind = ranking.index(i)
        f_inds.append(f_ind)
        f_names = np.append( f_names, names[f_ind] )
        Xin = Xw[:,f_inds]
        score = runModel(model, Xin, y, cv)
        f_scorelist.append( (f_names, score) )

    return(f_scorelist)


# In[6]:


"""
rankRFE runs the RFE feature elimination algorithm on the list of models provided by the models variable to provide 
a feature ranking for each which then is utilized by RFAperf to produce feature augmentation performance results. 
The resulting data is compiled into the modelsData variable which can then be passed onto a plotting function.
"""
def rankRFE(models, Xversions, y, names):
    lnames = len(names)
    FAstr = 'RFE'
    
    modelsData = []
    results = pd.DataFrame([], index=range(1,lnames+1))
    for inputType, Xw in Xversions:
        for model in models:
            modelname = str(model).partition('(')[0]
            rfe = RFE(model, 1)
            # rank RFE results
            rfe.fit(Xw, y)
            ranking = rfe.ranking_
            f_scorelist = RFAperf(ranking, str(model), Xw, y, names, cv=True)
            modelsData.append( (inputType, str(model), FAstr, ranking, f_scorelist) ) 
            f_ranking = [n for r, n in sorted( zip( ranking, names ) )]
            results[modelname[0:3] + FAstr + '-' + inputType[0:2]] = f_ranking
    
    return(modelsData, results)


# In[7]:


"""
plotRFAdata extracts the information provided in the modelsData variable which is compiled by running the RFAperf function over many different models (an instance of which is the rankRFE function above utilizing the RFE method) and plots the score curve for each model/test case.
"""
def plotRFAdata(modelsData, names):
    n = len(modelsData)
    l = len(names)
    
    fig = plt.figure()
    xvals = range(1,l+1)
    colorVec = ['ro', 'go', 'bo', 'co', 'mo', 'yo', 'ko', 'rs', 'gs', 'bs', 'cs', 'ms', 'ys', 'ks']
    for i in range(n):
        modelData = modelsData[i]
        inputType = modelData[0]
        modelstr = modelData[1]
        modelname = modelstr.partition('(')[0]
        FAstr = modelData[2]
        ranking = modelData[3]
        f_scorelist = modelData[4]
        f = np.array(f_scorelist)[:,0]
        s = np.array(f_scorelist)[:,1]
        labelstr = modelname[0:3] + FAstr + '-' + inputType[0:2]
        plt.plot(xvals, s, colorVec[i]+'-',  label=labelstr)
      
    fig.suptitle('Recursive Feature Augmentation Performance')
    plt.ylabel('R^2')
    #plt.ylim(ymax=1)
    plt.xlabel('Number of Features')
    plt.xlim(1-0.1,l+0.1)
    plt.legend(loc='lower right', fontsize=10)
    ax = fig.add_subplot(111)
    ax.set_xticks(xvals)
    plt.show()


# In[9]:


from sklearn.svm import SVR
from sklearn.feature_selection import RFE
import time

start = time.clock()
Smodels = [SVR(kernel="linear")]
Xversions = [('original', X), ('rescaled', rescaledX), ('standardized', standardizedX)]
modelsData, results = rankRFE(Smodels, Xversions, y, names)

end = time.clock()

print('Runtime was: ', end-start, 'seconds')


# In[ ]:


display(results)
plotRFAdata(modelsData, names)

