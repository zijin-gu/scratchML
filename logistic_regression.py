# Logistic Regression
import numpy as np

class LogisticRegression:
	def __init__(self, epoch, lr):

		self.epoch = epoch
		self.lr = lr

	def fit(self, X, y):

		self.n_samples, self.n_features = X.shape
		self.X = X
		self.y = y
		self.W = np.zeros(n_features)
		self.b = 0

		for i in range(self.epoch):
			self.update_weights()

		return self

	def update_weights(self):

		y_pred = self.predict(self.X)

		grad_W = -1 * self.X.T * (self.y - y_pred) / self.n_samples
		grad_b = -1 * np.sum(self.y - y_pred) / self.n_samples

		self.W = self.W - self.lr * grad_W
		self.b = self.b - self.lr * grad_b

		return self

	def sigmoid(self, z):

		return 1 / (1 + np.exp(-1 * z))


	def predict(self, X):

		z = X.dot(self.W) + self.b
		p = self.sigmoid(z)

		return np.where(p < 0.5, 0, 1)
