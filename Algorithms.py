import numpy as np

#PCA
class PCAModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.explained_variance = eigenvalues[sorted_idx][:self.n_components]
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def predict(self, X):
        if self.mean is None or self.components is None:
            raise ValueError("The PCA model has not been fitted yet.")

        X_centered = X - self.mean
        
        return np.dot(X_centered, self.components)
    
#KNN with PCA
class KNeighborsClassifier:
    
    def __init__(self, k=5, dims = 100):
        # dims = no. of reduced dimensions for PCA,
        # k = no. of nearest neighbours.
        self.k = k
        self.X_train = None
        self.y_train = None
        self.n_classes = 0
        self.dims = dims
        self.pca = PCAModel(n_components=self.dims)

    #Recording the training points
    def fit(self, X, y):
        self.pca.fit(X)
        X = self.pca.predict(X)
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.n_classes = len(np.unique(self.y_train))

    def predict_proba(self, X):
        X = np.asarray(self.pca.predict(X))
        n_samples = X.shape[0]

        #Distance calculation
        x_sq = np.sum(X**2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train.T**2, axis=0, keepdims=True)
        two_ab = -2 * np.dot(X, self.X_train.T)
        dists_sq = x_sq + two_ab + train_sq
        
        #Assigning probability to each class based on K nearest neighbours
        k_nearest_indices = np.argsort(dists_sq, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        probabilities = np.zeros((n_samples, self.n_classes), dtype=float)
        for c in range(self.n_classes):
            probabilities[:, c] = np.sum(k_nearest_labels == c, axis=1)
        probabilities = probabilities / self.k
                
        return probabilities

    #Returning class with the greatst probability
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
#Multiclass XGB by making some changes to our XGB for binary classification

class Node:
    def __init__(self, feat=None, threshold=None, left=None, right=None, value=None):
        self.feat = feat
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class XGBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_score = 0.0
        self.proba_threshold = 0.0

    #Softmax
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    #to one hot
    def _to_one_hot(self, y):
        self.n_classes = len(np.unique(y))
        one_hot = np.zeros((y.shape[0], self.n_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot

    #build tree function
    def _build_tree(self, X, grad, hess, depth):
        #terminal condition
        if depth >= self.max_depth or X.shape[0] <= 1:
            leaf_value = -np.sum(grad) / (np.sum(hess) + 1e-8)
            return Node(value=leaf_value)
        #finding best split
        best_gain = -float('inf')
        best_feat = None
        best_thresh = None
        best_left = None
        best_right = None
        G_total, H_total = np.sum(grad), np.sum(hess)
        for j in range(X.shape[1]):
            thresholds = np.unique(X[:, j])
            for threshold in thresholds:
                left = X[:, j] <= threshold
                right = ~left
                if np.any(left) and np.any(right):
                    G_l, H_l = np.sum(grad[left]), np.sum(hess[left])
                    G_r, H_r = np.sum(grad[right]), np.sum(hess[right])
                    gain = 0.5 * (
                        G_l**2/(H_l+1e-8) +
                        G_r**2/(H_r+1e-8) -
                        G_total**2/(H_total+1e-8)
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = j
                        best_thresh = threshold
                        best_left, best_right = left, right
        #if no split is possible
        if best_gain == -float('inf'):
            leaf_value = -np.sum(grad) / (np.sum(hess) + 1e-8)
            return Node(value=leaf_value)
        #recursion step
        left_node = self._build_tree(X[best_left], grad[best_left], hess[best_left], depth+1)
        right_node = self._build_tree(X[best_right], grad[best_right], hess[best_right], depth+1)
        return Node(feat=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    #prediction for a single row
    def _predict_row(self, x, node):
        while node.value is None:
            if x[node.feat] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    #fit function
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_one_hot = self._to_one_hot(y)
        
        pred = np.zeros((X.shape[0], self.n_classes), dtype=float)
        
        self.trees = []
        for _ in range(self.n_estimators):
            prob = self._softmax(pred)
            
            grad = prob - y_one_hot
            hess = prob * (1 - prob)
            
            round_trees = []
            
            # Build one tree for each class
            for k in range(self.n_classes):
                grad_k = grad[:, k]
                hess_k = hess[:, k]
                tree = self._build_tree(X, grad_k, hess_k, depth=0)
                round_trees.append(tree)
                update = np.array([self._predict_row(row, tree) for row in X])
                pred[:, k] += self.learning_rate * update
            self.trees.append(round_trees)

    #predict probability for each class
    def predict_proba(self, X):
        X = np.asarray(X)
        pred = np.zeros((X.shape[0], self.n_classes), dtype=float)
        
        for tree_group in self.trees:
            for k in range(self.n_classes):
                tree_k = tree_group[k]
                update = np.array([self._predict_row(row, tree_k) for row in X])
                pred[:, k] += self.learning_rate * update
        return self._softmax(pred)

    #Predicting the class with highest probability
    def predict(self, X):
        proba = self.predict_proba(X)

        return np.argmax(proba, axis=1)

if __name__ == '__main__':
    main()
