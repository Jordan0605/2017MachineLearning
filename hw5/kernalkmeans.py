from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from math import exp, sqrt

def dist(a, b, ax=1):
    return np.linalg.norm(a-b, axis=ax)

def dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def rbf(a, sigma=3):
    n = len(a)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i][j] = exp(-dist2(a[i], a[j])/(2*sigma**2))
    return G

if __name__ == "__main__":
    path = '/home/jordan/Documents/ml/hw5/homework#5/'
    filename = argv[1]
    k = int(argv[2])
    output = '/home/jordan/Documents/ml/hw5/output2/'
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    x = []
    y = []

    with open(path+filename, 'r') as f:
        for line in f:
            tmp = line.split()
            x.append(float(tmp[0]))
            y.append(float(tmp[1]))
    X = np.array(list(zip(x, y)))
    plt.scatter(x, y, c="black", s=7)
    plt.savefig(output+'0.png')
    
    X_kernel = rbf(X)
    idx = np.random.randint(len(X_kernel), size=k)
    C = X_kernel[idx, :]
    C_old = np.zeros(C.shape)
    C_low = np.zeros((k, 2))
    clusters = np.zeros(len(X_kernel))
    error = dist(C, C_old, None)
    
    iterator = 0
    while error != 0:
        iterator += 1
        for i in range(len(X_kernel)):
            distances = dist(X_kernel[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster

        C_old = deepcopy(C)
        fig, ax = plt.subplots()
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            points_kernel = [X_kernel[j] for j in range(len(X_kernel)) if clusters[j] == i]
            C[i] = np.mean(points_kernel, axis=0)
            points = np.array(points)
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
            C_low = np.mean(points, axis=0)
            ax.scatter(C_low[0], C_low[1], marker='*', s=200, c='black')
        error = dist(C, C_old, None)
        plt.savefig(output+str(iterator)+'.png')
