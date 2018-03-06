import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score

path = '/home/jordan/Documents/ml/hw6/data/'
colors = ['red', 'blue', 'green', 'purple', 'orange']

def load_data(path):
    X_train = np.genfromtxt(path+'X_train.csv', delimiter=',')
    y_train = np.genfromtxt(path+'T_train.csv', delimiter=',')
    X_test = np.genfromtxt(path+'X_test.csv', delimiter=',')
    y_test = np.genfromtxt(path+'T_test.csv', delimiter=',')
    return X_train, y_train, X_test, y_test

def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min()-1, x.max()+1
    y_min, y_max = y.min()-1, y.max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_decision_boundary(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8, **params)

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(path)
    C = 5
    #compare kernels and parameters
    clfs = (svm.SVC(kernel='linear', C=C),
            svm.LinearSVC(C=C),
            svm.SVC(kernel='rbf', gamma=0.05, C=C),
            svm.SVC(kernel='rbf', gamma=0.5, C=3),
            svm.SVC(kernel='rbf'),
            svm.SVC(kernel='poly', degree=2))
    models = (clf.fit(X_train, y_train) for clf in clfs)
    titles = ('Linear kernel', 
              'LinearSVC (linear kernel)',
              'RBF kernel (gamma=0.05, C=5)',
              'RBF kernel (gamma=0.5, C=3)',
              'RBF kernel (default)',
              'Polynomial (degree 2)')
    arr = []
    for model, title, clf in zip(models, titles, clfs):
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        arr.append([title, scores.mean(), model.score(X_test, y_test)])
    df = pd.DataFrame(arr, columns=["model", "5CV accuracy", "test accuracy"])
    print df
    
    #svm after pca
    clfs = (svm.SVC(kernel='rbf', gamma=0.05, C=5),
            svm.SVC(kernel='rbf', gamma=0.5, C=3),
            svm.SVC(kernel='rbf'))
    pca = PCA(n_components=2).fit(X_train, y_train)
    X_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    models = (clf.fit(X_pca, y_train) for clf in clfs)
    titles = ('RBF kernel (gamma=0.05, C=5)',
              'RBF kernel (gamma=0.5, C=3)',
              'RBF kernel (default)')
    arr = []
    print "SVM after pca:"
    for model, title, clf in zip(models, titles, clfs):
        scores = cross_val_score(clf, X_pca, y_train, cv=5)
        arr.append([title, scores.mean(), model.score(X_test_pca, y_test)])
    df = pd.DataFrame(arr, columns=["model", "5CV accuracy", "test accuracy"])
    print df
    
    #plot
    model = models[2]
    fig, ax = plt.subplots()
    xx, yy = make_meshgrid(X_pca[:, 0], X_pca[:, 1])
    plot_decision_boundary(ax, model, xx, yy)
    for i in range(len(y_train)):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], s=3, c=colors[int(y_train[i])-1])


    plt.savefig('decision_boundary.png')
