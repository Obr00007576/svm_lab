from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.1, 5, 100)
y = np.log(x) + np.random.normal(x, 0.3)
x = x.reshape(-1, 1)

epsilon_values = np.logspace(-2, 2, 50)
mse_values = []

for epsilon in epsilon_values:
    model = SVR(kernel='rbf', C=1, epsilon=epsilon)
    model.fit(x, y)
    y_pred = model.predict(x)
    mse = np.mean((y - y_pred) ** 2)
    mse_values.append(mse)
    
plt.plot(epsilon_values, mse_values)
plt.xscale('log')
plt.xlabel('Epsilon')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Epsilon')
plt.show()
'''
svm = SVR(kernel='rbf', epsilon=0.25, C=1)
x = x.reshape(-1, 1)
svm.fit(x, y)
y_pred = svm.predict(x)
plt.plot(x,y_pred)
plt.show()
'''