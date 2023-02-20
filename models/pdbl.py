import numpy as np
from sklearn.decomposition import PCA

class PDBLNet :
    def __init__(self, isPCA, n_components, scale):
        self.scale = scale
        self.isPCA = isPCA
        self.pca = PCA(n_components = n_components, whiten = False)

    def pinv(self, A, scale):
        return np.mat(scale * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)

    def fit(self, X, y):
        if self.isPCA :
            self.pca.fit(X)
            X = self.pca.transform(X)

        self.A = X
        self.y = y
        self.A_add = self.pinv(self.A, self.scale)
        self.W_PDBL = self.A_add.dot(self.y)

    def predict(self, X):
        if self.isPCA :
            X = self.pca.transform(X)

        A = X.copy()
        output = np.array(np.squeeze(A.dot(np.squeeze(self.W_PDBL))))

        return output

if __name__ == '__main__':
    pdblnet = PDBLNet(True, 20, 0.5)
    X = np.random.rand(100, 100)
    y = np.ones(100)
    pdblnet.fit(X, y)