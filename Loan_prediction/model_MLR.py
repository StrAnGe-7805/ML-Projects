import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

x_train = df_train.iloc[:,6:-2]
x_test = df_test.iloc[:,6:-1]

y_train = df_train.iloc[:,-1]
y_train = np.array(pd.get_dummies(y_train).values[:,0]).reshape(len(pd.get_dummies(y_train).values[:,0]),1)

gender_train = pd.get_dummies(df_train.iloc[:,1])
gender_test = pd.get_dummies(df_test.iloc[:,1])

married_train = pd.get_dummies(df_train.iloc[:,2])
married_test = pd.get_dummies(df_test.iloc[:,2])

dependents_train = df_train.iloc[:,3].values
dependents_test = df_test.iloc[:,3].values
for i in range(len(dependents_train)):
    if(dependents_train[i] == '3+'):
        dependents_train[i] = '3'
for i in range(len(dependents_test)):
    if(dependents_test[i] == '3+'):
        dependents_test[i] = '3'
dependents_train = np.array(dependents_train).reshape(len(dependents_train),1)
dependents_test = np.array(dependents_test).reshape(len(dependents_test),1)

educated_train = pd.get_dummies(df_train.iloc[:,4])
educated_test = pd.get_dummies(df_test.iloc[:,4])

employed_train = pd.get_dummies(df_train.iloc[:,5])
employed_test = pd.get_dummies(df_test.iloc[:,5])

area_train = np.array(pd.get_dummies(df_train.iloc[:,-2]).values[:,:-1])
area_test = np.array(pd.get_dummies(df_test.iloc[:,-1]).values[:,:-1])

from sklearn.impute import SimpleImputer
imputer_median = SimpleImputer(missing_values=np.nan,strategy = 'median')
imputer_mean = SimpleImputer(missing_values=np.nan,strategy = 'mean')

x_train[['Credit_History']] = imputer_median.fit_transform(x_train[['Credit_History']])
x_test[['Credit_History']] = imputer_median.fit_transform(x_test[['Credit_History']])
x_train[['Loan_Amount_Term']] = imputer_median.fit_transform(x_train[['Loan_Amount_Term']])
x_test[['Loan_Amount_Term']] = imputer_median.fit_transform(x_test[['Loan_Amount_Term']])
x_train[['LoanAmount']] = imputer_mean.fit_transform(x_train[['LoanAmount']])
x_test[['LoanAmount']] = imputer_mean.fit_transform(x_test[['LoanAmount']])

gender_train[['Male']] = imputer_median.fit_transform(gender_train[['Male']])
gender_train = np.array(gender_train.values[:,1]).reshape(len(gender_train.values[:,1]),1)
gender_test[['Male']] = imputer_median.fit_transform(gender_test[['Male']])
gender_test = np.array(gender_test.values[:,1]).reshape(len(gender_test.values[:,1]),1)

married_train[['Yes']] = imputer_median.fit_transform(married_train[['Yes']])
married_train = np.array(married_train.values[:,1]).reshape(len(married_train.values[:,1]),1)
married_test[['Yes']] = imputer_median.fit_transform(married_test[['Yes']])
married_test = np.array(married_test.values[:,1]).reshape(len(married_test.values[:,1]),1)

dependents_train = imputer_median.fit_transform(dependents_train)
dependents_test = imputer_median.fit_transform(dependents_test)

educated_train[['Graduate']] = imputer_median.fit_transform(educated_train[['Graduate']])
educated_train = np.array(educated_train.values[:,0]).reshape(len(educated_train.values[:,0]),1)
educated_test[['Graduate']] = imputer_median.fit_transform(educated_test[['Graduate']])
educated_test = np.array(educated_test.values[:,0]).reshape(len(educated_test.values[:,0]),1)

employed_train[['Yes']] = imputer_median.fit_transform(employed_train[['Yes']])
employed_train = np.array(employed_train.values[:,1]).reshape(len(employed_train.values[:,1]),1)
employed_test[['Yes']] = imputer_median.fit_transform(employed_test[['Yes']])
employed_test = np.array(employed_test.values[:,1]).reshape(len(employed_test.values[:,1]),1)

x_train = x_train.values
x_test = x_test.values

x_train = np.append(arr = x_train,values = area_train,axis = 1)
x_test = np.append(arr = x_test,values = area_test,axis = 1)
x_train = np.append(arr = employed_train,values = x_train,axis = 1)
x_test = np.append(arr = employed_test,values = x_test,axis = 1)
x_train = np.append(arr = educated_train,values = x_train,axis = 1)
x_test = np.append(arr = educated_test,values = x_test,axis = 1)
x_train = np.append(arr = dependents_train,values = x_train,axis = 1)
x_test = np.append(arr = dependents_test,values = x_test,axis = 1)
x_train = np.append(arr = married_train,values = x_train,axis = 1)
x_test = np.append(arr = married_test,values = x_test,axis = 1)
x_train = np.append(arr = gender_train,values = x_train,axis = 1)
x_test = np.append(arr = gender_test,values = x_test,axis = 1)

x_train_train = []
x_train_test = []
y_train_train = []
y_train_test = []
for i in range(int(len(x_train)/2),len(x_train)):
    if(i%21 == 0 or i%23 == 0):
        x_train_test.append(x_train[i])
        y_train_test.append(y_train[i][0])
    else:
        x_train_train.append(x_train[i])
        y_train_train.append(y_train[i][0])
for i in range(int(len(x_train)/2)):
    if(i%21 == 0 or i%23 == 0):
        x_train_test.append(x_train[i])
        y_train_test.append(y_train[i][0])
    else:
        x_train_train.append(x_train[i])
        y_train_train.append(y_train[i][0])
x_train_train = np.array(x_train_train)
x_train_test = np.array(x_train_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train_train = sc.fit_transform(x_train_train)
x_train_test = sc.transform(x_train_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy')
classifier.fit(x_train_train,y_train_train)

y_pred = classifier.predict(x_train_test)

cc = 0
cw = 0
wc = 0
ww = 0
c_val = []
for i in range(len(y_pred)):
    if(y_pred[i] > 0.5):
        if(y_train_test[i] == 1):
            cc  = cc + 1
        else:
            wc = wc + 1
    else:
        if(y_train_test[i] == 1):
            cw = cw + 1
        else:
            ww = ww + 1

import sys
np.set_printoptions(threshold=sys.maxsize)
sys.stdout=open("results_MLR.txt","w")
print(cc , cw , wc , ww)
print("Total correct predictions : ", (cc + ww))
print("Total wrong predictions : ",(cw+wc))