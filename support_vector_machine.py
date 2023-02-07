# Support Vector Machine
import numpy as numpy

# loss = \lambda ||w||^2 + \sum_{i=1}^{n} max{0, 1-y_i(W^TX + b)}
class SVM:

	def __init__(self, lambda_param, lr, epoch):
		self.lambda_param = lambda_param
		self.lr = lr
		self.epoch = epoch
		self.W = 0
		self.b = 0

	def fit(self, X, y):
		n_samples, n_features = X.shape
		reg = 0.5 * W * W

		for k in range(self.epoch):
			for i in range(n_samples):
				condition = y[i] * (np.dot(self.W, X[i]) + self.b)
				if condition >= 1:
					grad_W = 2 * self.lambda_param * self.W
					grad_b = 0
				else:
					grad_W = 2 * self.lambda_param * self.W - np.dot(X[i], y[i])
					grad_b = y[i]

				self.W -= self.lr * grad_W
				self.b -= self.lr * grad_b

	def predict(self, X):
		return np.sign(np.dot(self.W, X) + self.b)
