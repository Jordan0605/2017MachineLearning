from sys import argv
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import laplacian
from math import sqrt, exp

def dist(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def rbf(a, sigma=3):
    n = len(a)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i][j] = exp(-dist(a[i], a[j])/(2*sigma**2))
    return G


if __name__ == "__main__":
    filename = argv[1]
    k = int(argv[2])
    path = '/home/jordan/Documents/ml/hw5/homework#5/'
    output = '/home/jordan/Documents/ml/hw5/output3/'
    colors = ['r', 'g', 'b', 'y', 'c', 'm']

    x = []
    y = []

    with open(path+filename, 'r') as f:
        for line in f:
            tmp = line.split()
            x.append(float(tmp[0]))
            y.append(float(tmp[1]))
    X = np.array(list(zip(x, y)))
    
    X_kernel = rbf(X)
    L = laplacian(X_kernel, normed=True)
    w, v = np.linalg.eig(L)
    U = v[:,:k]
    T = np.zeros(U.shape)
    for i in range(len(U)):
        for j in range(k):
            T[i][j] = U[i][j]/sqrt(reduce(lambda a, b: a**2 + b**2, U[i]))
    
    kmeans = KMeans(n_clusters=k).fit(T)
    pred = kmeans.labels_
    
    for i in range(k):
        points = [X[j] for j in range(len(X)) if pred[j] == i]
        points = np.array(points)
        plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    plt.savefig(output+'result.png')
