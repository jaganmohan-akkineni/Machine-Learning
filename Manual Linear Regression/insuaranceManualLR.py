import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt


data = pd.read_csv('insurance.csv')

theta = [0, 0]

# store theta values after each iteration in 3-fold validation
theta0, theta1 = [], []

alpha = 0.001
iterations = 20000

X = data['X']
Y = data['Y']

# Mean squared error cost function
def cost(x, y):
    cost = 0
    for i in range(len(x)):
        cost = cost + (theta[0] + theta[1]*x[i] - y[i])**2
    cost = cost/(len(x))
    return cost

def gradientDescent(x, y):
    sum1 = 0
    sum2 = 0
    for i in range(len(x)):
        sum1 = sum1 + (theta[0] + theta[1]*x[i] - y[i])
        sum2 = sum2 + (theta[0] + theta[1]*x[i] - y[i])*x[i]

    theta[0] = theta[0] - (alpha) * (1/len(x)) * sum1
    theta[1] = theta[1] - (alpha) * (1/len(x)) * sum2

def linearRegression(x, y, iters):
    for i in range(iters):
        gradientDescent(x, y)

# specifying no of folds for cross validation
kf = KFold(n_splits=3) # Define the split - into 2 folds 

total_cost  = 0
#data = shuffle(data)

# 3-fold cross validation
for train_index, test_index in kf.split(data):

    # splitting data into test data and train data
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # initialize theta values to zero in every iteration
    theta[0], theta[1] = 0, 0

    # Manual Linear Regression
    linearRegression(X_train.tolist(), Y_train.tolist(), iterations)
    theta0.append(theta[0])
    theta1.append(theta[1])
    total_cost = total_cost + cost(X_test.tolist(), Y_test.tolist())

    
# Mean of errors
total_cost = total_cost/3

print('\nManual linear regression:')
print('\nTheta0:', theta0)
print('Theta1:', theta1)

print('\nAvg of Theta0:', np.mean(theta0), '\nAvg of Theta1', np.mean(theta1))  

print('MSE of Manual LR:', total_cost, '\n')

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(data[['X']], data['Y'], test_size=0.2)

# Fitting the model
linreg = LinearRegression(normalize=True)
linreg.fit(X_train, y_train)

print('Linear Regression using sciki-learn package:')
print("\nCoefficients:", list(zip(['X'], linreg.coef_)))
print("Intercept:", linreg.intercept_)

y_pred = linreg.predict(X_test)
print("MSE:", metrics.mean_squared_error(y_test, y_pred), '\n')

#plot data
plt.scatter(data.X, data.Y)
plt.title('Motor Insurance in Sweden')
plt.xlabel('Number of claims', color='#1C2833')
plt.ylabel('Total payment(in 1000s)', color='#1C2833')

#plotting manual linear regression line
x = np.linspace(-1,130,100)
y = (np.mean(theta1))*x+np.mean(theta0)
plt.plot(x, y, '-r', label='manual')

#plotting linear regression line from scikit learn
x = np.linspace(-1,130,100)
y = (linreg.coef_)*x+linreg.intercept_
plt.plot(x, y, '-g', label='scikit-learn')

plt.legend()

plt.show()