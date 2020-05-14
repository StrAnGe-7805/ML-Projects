import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")

x = df.iloc[:,:-1].values
y = df.iloc[:,-1]
y = pd.get_dummies(y)
y = y.values

x_train = []
x_test = []
y_train = []
y_test = []

for i in range(int(len(x)/2),len(x)):
    if(i%21 == 0 or i%23 == 0):
        x_test.append(x[i])
        y_test.append(y[i])
    else:
        x_train.append(x[i])
        y_train.append(y[i])

for i in range(int(len(x)/2)):
    if(i%21 == 0 or i%23 == 0):
        x_test.append(x[i])
        y_test.append(y[i])
    else:
        x_train.append(x[i])
        y_train.append(y[i])

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# x_train = x[:-2,:]
# x_test = x[-2:,:]
# y_train = y[:-2,:]
# y_test = y[-2:,:]

# print(x_test)
# print(y_test)

# from sklearn.model_selection import train_test_split
# x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 4))
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 3,init = 'uniform',activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

classifier.fit(x_train,y_train,batch_size = 10,nb_epoch = 100)

y_pred = classifier.predict(x_test)
for x in y_pred:
    if x[0] > x[1] and x[0] > x[2]:
        x[0] = 1
        x[1] = 0
        x[2] = 0
    elif x[1] > x[0] and x[1] > x[2]:
        x[0] = 0
        x[1] = 1
        x[2] = 0
    elif x[2] > x[0] and x[2] > x[1]:
        x[0] = 0
        x[1] = 0
        x[2] = 1

y_pred = y_pred.astype(int)

c = 0
w = 0

for i in range(len(y_pred)):
    if(y_pred[i][0] == y_test[i][0] and y_pred[i][1] == y_test[i][1] and y_pred[i][2] == y_test[i][2]):
        c = c + 1
    else:
        w = w + 1
print("Correct predictions are :  " ,c,"   out of :  ",(c+w))
print("Wrong predictions are :  " ,w,"   out of :  ",(c+w))
print("Acccuracy is :" ,((c/(c+w))*100),"%")