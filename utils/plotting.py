# plotting functions
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def plot_regression_data(X, y):
    fig = plt.figure()
    if X.shape[1] == 1:
        plt.scatter(X, y)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs = X[:, 0], ys = X[:, 1], zs=y)
    plt.show()
    
def plot_regression(X, y, y_pred):
    fig = plt.figure()
    if X.shape[1] == 1:
        plt.scatter(X, y)
        plt.plot(X, y_pred)
        plt.show()
        
def plot_classification_data(X, y):
    fig = plt.figure()
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c = y)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs = X[:, 0], ys = X[:, 1], zs=X[:, 2], c = y)
    plt.show()

def plot_classification(X, y, classifier):
    '''reg is an instance of a custom classification class with .predict method'''
    n_classes = len(np.unique(y))
    f, ax = plt.subplots()
    n_min = min(np.min(X[:, 0]), np.min(X[:, 1])) - 1
    n_max = max(np.max(X[:, 0]), np.max(X[:, 1])) + 1
    x0, x1 = np.mgrid[n_min:n_max:.01, n_min:n_max:.01]
    grid = np.c_[x0.ravel(), x1.ravel()]
    pred = classifier.predict(grid).reshape(x0.shape)
    ax.contourf(x0, x1, pred, 25, vmin=0, vmax=n_classes - 1)
    ax.scatter(X[:, 0], X[:, 1], c = y, vmin=0, vmax=n_classes - 1, edgecolor="white")
    ax.set(aspect="equal", xlim=(n_min, n_max), ylim=(n_min, n_max), xlabel="$X_1$", ylabel="$X_2$")
    plt.show()
    
def plot_loss(loss):
    f, ax = plt.subplots()
    ax.plot(np.arange(1,len(loss)+1), loss)
    plt.title('Loss')
    ax.set(xlabel = 'epoch', ylabel = 'loss')
    plt.show()
    
def plot_clustering(X, y = None):
    fig = plt.figure()
    if y is None:
        y = ['blue']*X.shape[0]
    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c = y)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs = X[:, 0], ys = X[:, 1], zs=X[:, 2], c = y)
    plt.show()