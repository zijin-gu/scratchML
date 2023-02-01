# k Nearest Neighbors
import numpy as np
from collections import Counter

# kNN classifier (regressor)
def kNNClassifier:

	def __init__(self, k):

		self.k = k

	def fit(self, X_train, y_train):

		self.X_train = X_train
		self.y_train = y_train
		self.n_train = X_train.shape[0]

		return self

	def euclidean_distance(self, x1, x2):

		return np.sqrt(np.sum((x1 - x2) ** 2))

	def predict(self, X_test):

		self.X_test = X_test
		self.n_test = X.X_test.shape[0]
		y_pred = np.zeros(self.n_test)

		for i, test_sample in enumerate(X_test):
			neighbors = self.find_neighbors(test_sample)
			y_pred[i] = self.vote(neighbors) # np.mean(neighbors) if kNN regressor

		return y_pred

	def find_neighbors(self, x):

		distances = np.zeros(self.n_train)
		for i in range(self.n_train):
			distances[i] = self.euclidean_distance(test_sample, self.X_train[i])

		sort_indices = np.argsort(distances)[:self.k]

		return self.y_train[sort_indices]

	def vote(self, y_neighbors):

		return Counter(y_neighbors).most_common(1)[0][0]
