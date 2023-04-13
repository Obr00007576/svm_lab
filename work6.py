from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel()

# Add some noise
y[::4] += 3 * (0.5 - np.random.rand(X.shape[0] // 4))

# Train the model
epsilon_values = np.logspace(-2, 2, 50)
mse_values = []

for epsilon in epsilon_values:
    model = SVR(kernel='rbf', C=1, epsilon=epsilon)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    mse_values.append(mse)
    #plt.plot(X,y_pred)
    #plt.show()


# Plot the results
plt.plot(epsilon_values, mse_values)
plt.xscale('log')
plt.xlabel('Epsilon')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Epsilon')
plt.show()