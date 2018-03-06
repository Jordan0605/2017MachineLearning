from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

def dist(a, b, ax=1):
    return np.linalg.norm(a-b, axis=ax)


if __name__ == "__main__":
    path = '/home/jordan/Documents/ml/hw5/homework#5/'
    filename = argv[1]
    k = int(argv[2])
    output = '/home/jordan/Documents/ml/hw5/output/'
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
    init_x = np.random.randint(np.min(X)+2, np.max(X)-2, size=k)
    init_y = np.random.randint(np.min(X)+2, np.max(X)-2, size=k)
    C = np.array(list(zip(init_x, init_y)), dtype=np.float32)
    plt.scatter(init_x, init_y, marker='*', s=200, c='g')
    plt.savefig(output+'0.png')

    C_old = np.zeros(C.shape)
    clusters = np.zeros(len(X))
    error = dist(C, C_old, None)
    
    iterator = 0
    while error != 0:
        iterator += 1
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster

        C_old = deepcopy(C)
        fig, ax = plt.subplots()
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
            points = np.array(points)
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        error = dist(C, C_old, None)
        ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='black')
        plt.savefig(output+str(iterator)+'.png')

