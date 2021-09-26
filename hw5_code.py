# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:19:59 2021

@author: ruchuan2
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns

################################ descriptive statistics (exploratory data analysis) ##########################

data = pd.read_csv("C:/Users/ruchuan2/Box/IE 517 Machine Learning in FIN Lab/HW5/hw5_treasury yield curve data.csv", header='infer')
del data['Date']

# del  data["Date"]  ## delete the date variable

# This dataset includes a measure of yield on each treasury bond instrument for each tenor of maturity from 1 year to 30 year. 
# (Attributes=30; SVENF01 - SVENF30).

#It also includes the daily adjusted close price of the PIMCO Total Return Bond fund (Ticker=PTTAX). This is the target variable.

data.head

### summary of the data
summary = data.describe()
print(summary)


## the distribution of target variable

PTTAX = data['Adj_Close']
plt.hist(PTTAX)
plt.xlabel('PIMCO Total Return Bond Fund')
plt.ylabel('count')
plt.show()

### show the distributrion of each bond's yield
### year 1, 2, 3, 4, 5, 10, 15, 20, 25, 30 as examples 

Yield = [data['SVENF01'],data['SVENF02'],data['SVENF03'],data['SVENF04'], data['SVENF05'],data['SVENF06'],data['SVENF07'],data['SVENF08'],data['SVENF09'],data['SVENF10'], 
         data['SVENF11'],data['SVENF12'],data['SVENF13'],data['SVENF14'], data['SVENF15'],data['SVENF16'],data['SVENF17'],data['SVENF18'],data['SVENF19'],data['SVENF20'], 
         data['SVENF21'],data['SVENF22'],data['SVENF23'],data['SVENF24'], data['SVENF25'],data['SVENF26'],data['SVENF27'],data['SVENF28'],data['SVENF29'],data['SVENF30']]

plt.boxplot(Yield)
plt.xlabel("Maturity")
plt.ylabel("Yield")
plt.show()

## correlation matrix

corr_matrix = np.corrcoef(data.values.T)
hm = sns.heatmap(corr_matrix,
                 cbar=True,
                 annot=False,
                 square=True,
                 yticklabels=['SVENF01', 'SVENF02', 'SVENF03', 'SVENF04', 'SVENF05', 'SVENF06', 'SVENF07', 'SVENF08', 'SVENF09', 'SVENF10',
                              'SVENF11', 'SVENF12', 'SVENF13', 'SVENF14', 'SVENF15', 'SVENF16', 'SVENF17', 'SVENF18', 'SVENF19', 'SVENF20',
                              'SVENF21', 'SVENF22', 'SVENF23', 'SVENF24', 'SVENF25', 'SVENF26', 'SVENF27', 'SVENF28', 'SVENF29', 'SVENF30'],
                 xticklabels=['SVENF01', 'SVENF02', 'SVENF03', 'SVENF04', 'SVENF05', 'SVENF06', 'SVENF07', 'SVENF08', 'SVENF09', 'SVENF10',
                              'SVENF11', 'SVENF12', 'SVENF13', 'SVENF14', 'SVENF15', 'SVENF16', 'SVENF17', 'SVENF18', 'SVENF19', 'SVENF20',
                              'SVENF21', 'SVENF22', 'SVENF23', 'SVENF24', 'SVENF25', 'SVENF26', 'SVENF27', 'SVENF28', 'SVENF29', 'SVENF30'])
plt.show()

################################ split the training and test sets ##############################

### standardlized
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
data = sc.fit_transform(data)
data = DataFrame(data)

X, y = data.iloc[:,0:30].values, data.iloc[:,30]
# Split the dataset into a training and a testing set

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print( X_train.shape, y_train.shape, X_test.shape, y_test.shape)
############################## Part2: PCA ##################################################

from sklearn.decomposition import PCA

### n_components=None ############

pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
var = pca.explained_variance_ratio_
cum_var = np.cumsum(var)


### n_components=3 ############

pca = PCA(n_components=3)
X_train_pca_n3 = pca.fit_transform(X_train)
X_test_pca_n3 = pca.transform(X_test)
var_n3 = pca.explained_variance_ratio_
cum_var_n3 = np.cumsum(var_n3)


###################################### part 3 Linear regression and SVM ###############################################

## original dataset #######
## Linear Regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_train_pred_linear = lr.predict(X_train)
y_test_pred_linear = lr.predict(X_test)

### RMSE

from sklearn.metrics import mean_squared_error
rmse_train_linear = math.sqrt(mean_squared_error(y_train, y_train_pred_linear))
rmse_test_linear = math.sqrt(mean_squared_error(y_test, y_test_pred_linear))

## calculate R-squre

from sklearn.metrics import r2_score
r_square_train_linear = r2_score(y_train, y_train_pred_linear)
r_square_test_linear = r2_score(y_test, y_test_pred_linear)

######### SVM

from sklearn import svm
svr = svm.SVR(kernel='linear')
svr.fit(X_train,y_train)
y_train_pred_svm = svr.predict(X_train)
y_test_pred_svm = svr.predict(X_test)

### RMSE

from sklearn.metrics import mean_squared_error
rmse_train_svm = math.sqrt(mean_squared_error(y_train, y_train_pred_svm))
rmse_test_svm = math.sqrt(mean_squared_error(y_test, y_test_pred_svm))

## calculate R-squre

from sklearn.metrics import r2_score
r_square_train_svm = r2_score(y_train, y_train_pred_svm)
r_square_test_svm = r2_score(y_test, y_test_pred_svm)



#########################################  PCA transformed dataset with 3PCs ########################################


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train_pca_n3, y_train)
y_train_pred_linear_pca = lr.predict(X_train_pca_n3)
y_test_pred_linear_pca = lr.predict(X_test_pca_n3)

### RMSE

from sklearn.metrics import mean_squared_error
rmse_train_linear_pca= math.sqrt(mean_squared_error(y_train, y_train_pred_linear_pca))
rmse_test_linear_pca = math.sqrt(mean_squared_error(y_test, y_test_pred_linear_pca))

## calculate R-squre

from sklearn.metrics import r2_score
r_square_train_linear_pca = r2_score(y_train, y_train_pred_linear_pca)
r_square_test_linear_pca = r2_score(y_test, y_test_pred_linear_pca)

######### SVM

from sklearn import svm
svr = svm.SVR(kernel='linear')
svr.fit(X_train_pca_n3, y_train)
y_train_pred_svm_pca = svr.predict(X_train_pca_n3)
y_test_pred_svm_pca = svr.predict(X_test_pca_n3)

### RMSE

from sklearn.metrics import mean_squared_error
rmse_train_svm_pca  = math.sqrt(mean_squared_error(y_train, y_train_pred_svm_pca))
rmse_test_svm_pca  = math.sqrt(mean_squared_error(y_test, y_test_pred_svm_pca))

## calculate R-squre

from sklearn.metrics import r2_score
r_square_train_svm_pca = r2_score(y_train, y_train_pred_svm_pca)
r_square_test_svm_pca = r2_score(y_test, y_test_pred_svm_pca)


print("My name is Richie Ma")
print("My NetID is: ruchuan2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")






















