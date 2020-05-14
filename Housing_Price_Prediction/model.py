import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

x , y = load_boston(return_X_y=True)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(np.array(y).reshape(-1,1))

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
y_pred = sc_y.inverse_transform(y_pred)
y_test = sc_y.inverse_transform(y_test)

for i in range(len(y_pred)):
    print(y_test[i] , "    ",y_pred[i])