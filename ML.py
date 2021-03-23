
"""
Please refer to me befroe using the given code or adding any changes in it.
@author: ISHAN
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

df = pd.read_csv (r'E:\Whatsapp data\whatsapp data\INSAID\PS\Churn.csv',na_values=['.'])


df=df.replace(to_replace ="No phone service", value ="No")
	
df=df.replace(to_replace ="No internet service",value ="No")

#A= ((df.iloc[:,20]))


le = LabelEncoder() 
#These constants are to quickly tweak the number of features you want
old=1
new=20
col_in=1
col_out=21

#Training and Validation Data separater parameter
m_train=5000
m_valid=2043

#Implementing LabelEncoder
for i in range(col_in,col_out):
    df.iloc[:,i]=le.fit_transform(df.iloc[:,i])
df.loc[0]=int(1) 

##############################################
#Loading Training and Validation Datasets
X_train=df.iloc[1:m_train,old:new]
Y_train=df.iloc[1:m_train,20]
X_valid=df.iloc[m_train+1:m_valid+m_train,old:new]
Y_valid=df.iloc[m_train+1:m_valid+m_train,20]

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, Y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),('logisticregression', LogisticRegression())])

LinReg=(pipe.score(X_valid, Y_valid))


##############################################
df.loc[0]=np.random.rand() #To randomize first row of matrix.

X_train=df.iloc[1:m_train,old:new]
Y_train=df.iloc[1:m_train,20]
X_valid=df.iloc[m_train+1:m_valid+m_train,old:new]
Y_valid=df.iloc[m_train+1:m_valid+m_train,20]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4,), random_state=10)
clf.fit(X_train, Y_train)
     
NP=(clf.score(X_valid,Y_valid))
##############################################
df.loc[0]=int(1)

X_train=df.iloc[1:m_train,old:new]
Y_train=df.iloc[1:m_train,20]
X_valid=df.iloc[m_train+1:m_valid+m_train,old:new]
Y_valid=df.iloc[m_train+1:m_valid+m_train,20]

pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
pipe.fit(X_train, Y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),('logisticregression', RandomForestRegressor())])
RFR=(pipe.score(X_valid, Y_valid))

###############################################
pipe = make_pipeline(StandardScaler(), SVR())
pipe.fit(X_train, Y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),('SVR', SVR())])

SVR=(pipe.score(X_valid, Y_valid))
###############################################
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, Y_train)
Pipeline(steps=[('standardscaler', StandardScaler()),('SVR', LinearRegression())])

Linear=(pipe.score(X_valid, Y_valid))
################################################

#To print the result:
print('Logistic Regression- ',LinReg,'\n Neural Network-  ',NP,'\n Random forrest Regression-  ',RFR, '\n SVR-  ',SVR, '\n Linear Regression-  ',Linear )
