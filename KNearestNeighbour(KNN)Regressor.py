import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\21st\EMP SAL.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Fitting KNN to the dataset

from sklearn.neighbors import KNeighborsRegressor
#regressor = KNeighborsRegressor()
regressor = KNeighborsRegressor(n_neighbors =4, weights = 'uniform',algorithm='auto')
#regressor = KNeighborsRegressor(n_neighbors =3, weights = 'uniform',algorithm='ball_tree')
#regressor = KNeighborsRegressor(n_neighbors =4, weights = 'uniform',algorithm='brute')
#regressor = KNeighborsRegressor(n_neighbors =3, weights = 'uniform',algorithm='kd_tree')

# In KNeighborsRegression when we change the n_neighbors =3,4,5,6 the values are change 
# when we change the algorithm and without chainging the n_neighbours values all the y_predictions are same corrsponding to the n_neighbours value                                     

regressor.fit(X,y)

regressor.fit(X,y)
# Predctions
y_pred = regressor.predict([[6.5]])
y_pred

# Visualising the Knn results
%matplotlib inline
plt.scatter(X,y, color ='red')
plt.plot(X,regressor.predict(X),color ='blue')
plt.title('Truth or Bluff (KNN)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color ='red')
plt.title('Truth or Bluff (KNN)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()