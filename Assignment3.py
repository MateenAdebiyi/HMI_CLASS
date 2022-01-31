import pandas as pd
#df = pd.read_csv("/Users/aaangd/Downloads/HMI-cancer-data.tsv" , sep = '\t')
df = pd.read_csv("HMI-cancer-data.tsv" , sep = '\t')
X= df.drop('Overall Survival (Months)' , axis =1)
y = df['Overall Survival (Months)']
y = y.fillna(y.median())
#X['Patient Status'].isnull().sum()
X['Patient Status'] = X['Patient Status'].fillna(X['Patient Status'].mode())
#X.loc[9834,['Patient Status']]= 'ALIVE'
#X.loc[9835,['Patient Status']]= 'ALIVE'
X['Fraction Genome Altered'] = X['Fraction Genome Altered'].fillna(X['Fraction Genome Altered'].median())
X['Mutation Count'] = X['Mutation Count'].fillna(X['Mutation Count'].median())
X['Sample coverage'] = X['Sample coverage'].fillna(X['Sample coverage'].median())
X['Tumor Purity'] = X['Tumor Purity'].fillna(X['Tumor Purity'].median())
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
X['Patient Status']= label_encoder.fit_transform(X['Patient Status'])
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

parameters = {
    'max_depth': range (2, 10, 1),
    
    'min_samples_split': range(10, 60, 10),
    'min_samples_leaf': range(1,5),
    #'bootstrap': [True],
    'max_features': ['auto', 'log2'],
    
    #'n_estimators': [5, 6, 7, 8, 9, 10, 11, 12, 13, 15]          
    
}


gs = GridSearchCV(model,
                  param_grid = parameters,
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error',verbose=True, refit=True)

gs.fit(X, y)

print('The best parameters for the DecisionTreeRegressor are' , gs.best_params_)
#print(-gs.best_score_)

from sklearn.ensemble import RandomForestRegressor

estimator = RandomForestRegressor()
param_grid = { 
            "n_estimators"      : [10,20,30],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "bootstrap": [True, False]
            }
grid = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)
grid.fit(X, y)
#print(grid.best_params_)
#print(-grid.best_score_)
print ('                                                               ')
print('The best parameters for the RandomForestRegressor are' , grid.best_params_)

from sklearn.linear_model import LinearRegression
import numpy as np
model = LinearRegression()
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(model,parameters, cv=5)
grid.fit(X, y)
#print ("r2 / variance : ", grid.best_score_)
#print("Residual sum of squares: %.2f"
              #% np.mean((grid.predict(X) - y) ** 2))
             
print ('                                                               ')
print('The best parameters for the LinearRegression are' , grid.best_params_)