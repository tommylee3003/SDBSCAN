from annoy import AnnoyIndex
import numpy as np
from scipy.sparse import csr_matrix


class fastkNN:
    def __init__(self, n_neighbors, n_trees=5, metric='euclidean', return_distance=False):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.metric = metric
        self.return_distance = return_distance

    def fit(self, X):

        model = AnnoyIndex(X.shape[1], self.metric)
        for i in range(X.shape[0]):
            model.add_item(i, X[i])

        model.build(self.n_trees)
        nn = []
        for i in range(X.shape[0]):
           nn.append(model.get_nns_by_item(i, self.n_neighbors + 1, search_k=-1, include_distances=self.return_distance))
        self._nn = np.asarray(nn)
        self._fit_X = X
        return self

    def kneighbors(self):
        nn = self._nn
        indptr = np.asarray(nn[:, 0, :][:, 1:], dtype=np.int64)

        if self.return_distance:
            distance = nn[:, 1, :][:, 1:]
            return (distance, indptr)
        else:
            return (indptr)

    def kneighbors_graph(self, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        n_samples = self._fit_X.shape[0]
        n_nonzero = n_samples * n_neighbors
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)
        nn = self._nn
        if not self.return_distance:
            A_data = np.ones(n_samples * n_neighbors, dtype=np.int8)
            A_ind = nn[:, 1: (n_neighbors+1)]
        else:
            A_data = nn[:, 1, :][:, 1: (n_neighbors+1)]
            A_data = np.ravel(A_data)
            A_ind = nn[:, 0, :][:, 1: (n_neighbors+1)]

        kneighbors_graph = csr_matrix((A_data, A_ind.ravel(), A_indptr), shape=(n_samples, n_samples))

        return kneighbors_graph