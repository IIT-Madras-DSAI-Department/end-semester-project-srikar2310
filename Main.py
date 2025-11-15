import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

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
    
class KNeighborsClassifier:
    
    def __init__(self, k=7, dims = 100):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.n_classes = 0
        self.dims = dims
        self.pca = PCAModel(n_components=self.dims)

    def fit(self, X, y):
        self.pca.fit(X)
        X = self.pca.predict(X)
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.n_classes = len(np.unique(self.y_train))

    def predict_proba(self, X):
        X = np.asarray(self.pca.predict(X))
        n_samples = X.shape[0]

        x_sq = np.sum(X**2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train.T**2, axis=0, keepdims=True)
        two_ab = -2 * np.dot(X, self.X_train.T)
        dists_sq = x_sq + two_ab + train_sq

        k_nearest_indices = np.argsort(dists_sq, axis=1)[:, :self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        probabilities = np.zeros((n_samples, self.n_classes), dtype=float)
        for c in range(self.n_classes):
            probabilities[:, c] = np.sum(k_nearest_labels == c, axis=1)

        probabilities = probabilities / self.k
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    

def read_data(trainfile='MNIST_train.csv', validationfile='MNIST_validation.csv'):
    
    dftrain = pd.read_csv(trainfile)
    dfval = pd.read_csv(validationfile)

    featurecols = list(dftrain.columns)
    featurecols.remove('label')
    featurecols.remove('even')
    targetcol = 'label'

    Xtrain = dftrain[featurecols]
    ytrain = dftrain[targetcol]
    
    Xval = dfval[featurecols]
    yval = dfval[targetcol]

    return (Xtrain, ytrain, Xval, yval)

#One vs All accuracy scores
def calculate_scores(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    overall_accuracy = np.mean(y_true == y_pred)
    unique_labels = np.unique(y_true)
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'class_metrics': {}
    }
    for class_label in unique_labels:
        is_class_true = (y_true == class_label)
        is_class_pred = (y_pred == class_label)
        
        TP = np.sum(is_class_true & is_class_pred)
        TN = np.sum(~is_class_true & ~is_class_pred)
        FP = np.sum(~is_class_true & is_class_pred)
        FN = np.sum(is_class_true & ~is_class_pred)
        
        total_negatives = TN + FP
        total_samples = len(y_true)

        fpr = FP / total_negatives if total_negatives > 0 else 0.0
        fnr = FN / total_negatives if total_negatives > 0 else 0.0
        acc = (TP + TN) / total_samples if total_samples > 0 else 0.0
        
        metrics['class_metrics'][str(class_label)] = {
            'accuracy': acc,
            'fpr': fpr,
            'fnr': fnr,
        }
        
    return metrics

Xtrain,ytrain,Xval,yval = read_data()

knn = KNeighborsClassifier(k=16, dims = 150)
knn.fit(Xtrain,ytrain)

ypred = knn.predict(Xval)

print(calculate_scores(yval, ypred))