# Linear Regression
import numpy as np
import copy

# (1) Optimize via gradient descent
class LinearRegression():

	def __init__(self, epoch, lr):

		self.epoch = epoch
		self.lr = lr

	def fit(self, X, y):

		self.n_samples, self.n_features = X.shape
		self.W = np.zeros(n_features)
		self.b = 0
		self.X = X
		self.y = y

		for i in range(self.epoch):
			self.update_weights()

		return self

	def update_weights(self):

		y_pred = self.predict(self.X)

		grad_W = -2 * self.X.T * (self.y - y_pred) / self.n_samples
		grad_b = -2 * (np.sum(self.y - y_pred)) / self.n_samples

		self.W = self.W - self.lr * grad_W
		self.b = self.b - self.lr * grad_b

		return self


	def predict(self, X):

		return X.dot(self.W) + self.b


# (2) Use ordinary least square (OLS): (X.T*X)^(-1)*X.T*y
class LinearRegression():

	def __init__(self):
		pass

	def fit(self, X, y):

		self.X = X
		self.y = y

		X = copy.deepcopy(self.X)
		dummy = np.ones(X.shape[0])
		X = np.concatenate((dummy, X), axis=1)

		beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
		self.W = beta[1:]
		self.b = beta[0]

		return self

	def predict(self, X):

		return X.dot(self.W) + self.b
