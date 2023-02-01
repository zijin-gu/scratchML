# K-means
import numpy as np

class KMeans:
	def __init__(self, n_clusters, epoch):

		self.n_clusters = n_clusters
		self.epoch = epoch

	def fit(self, X_train):

		self.X_train = X_train
		self.n_train = X_train.shape[0]
		self.n_features = X_train.shape[1]

		min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
		centroids = np.zeros([self.n_clusters, self.n_features])
		for i in range(self.n_clusters):
			centroids[i] = np.random.uniform(min_, max_)

		for i in range(self.epoch):
			centroids = self.update_centroids(centroids)
		self.centroids = centroids

		return self

	def euclidean_distance(self, x1, x2):

		return np.sqrt(np.sum((x1 - x2) ** 2))

	def update_centroids(self, centroids):

		clusters = np.zeros(self.n_train)
		for i, train_sample in enumerate(self.X_train):
			distances = np.zeros(self.n_clusters)
			for j, center in enumerate(centroids):
				distances[j] = self.euclidean_distance(train_sample, center)
			clusters[i] = np.argmin(distances)

		for i in range(self.n_clusters):
			points = self.X_train[clusters == i]
			centroids[i] = np.mean(points, axis=0)

		return centroids


	def predict(self, X_test):

		y_pred = np.zeros(X_test.shape[0])
		for i, test_sample in enumerate(X_test):
			distances = np.zeros(self.n_clusters)
			for j, center in enumerate(self.centroids):
				distances[j] = self.euclidean_distance(test_sample, center)
			y_pred[i] = np.argmin(distances)

		return y_pred

