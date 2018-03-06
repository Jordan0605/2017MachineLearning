from dataGenerator import *
from matrixCalculator import *
from sys import argv, float_info
from math import exp, log
from random import random
import numpy as np

def sigmoid(x, w):
    z_arr = []
    for i in range(len(x)):
        z = 0.0
        for j in range(len(w)):
            z += x[i][j]*w[j]
        z_arr.append(z)
    Z = z_arr
    for i in range(len(z_arr)):
        try:
            Z[i] = 1.0/(1.0+exp(-Z[i]))
        except OverflowError:
            Z[i] = float_info.min
    Z = map(lambda a: float_info.min if a < float_info.min else a, Z)
    Z = map(lambda a: 0.99999999999 if a > 0.99999999999 else a, Z)
    return Z

def log_likelihood(x, y, w):
    prob = sigmoid(x, w)
    sum_likelihood = 0.0
    for i in range(len(y)):
        sum_likelihood += (y[i]*log(prob[i]) + (1 - y[i])*log(1-prob[i]))
    return sum_likelihood

def gradient(x, y, w):
    prob = sigmoid(x, w)
    xt = np.transpose(x) #xt = transpose(x)
    g = []
    for i in range(len(y)):
        g.append([prob[i] - y[i]])
    return np.dot(xt, g) #return multiply(xt, g)

def hessian(x, y, w):
    prob = sigmoid(x, w)
    #xt = transpose(x)
    xt = np.transpose(x)
    n = len(x)
    D = [[0.0 for i in range(n)] for j in range(n)]
    for i in range(n):
        D[i][i] = prob[i]*(1 - prob[i])
    return np.dot(xt, D).dot(x) #return multiply(multiply(xt, D), x)
    
def newtons_method(x, y):
    w = [0.0, 0.0, 0.0]
    l = log_likelihood(x, y, w)
    while True:
        g = gradient(x, y, w)
        h = hessian(x, y, w)
        h_inv = np.linalg.inv(h) #h_inv = inverse(h)
        delta_w = np.dot(h_inv, g) #delta_w = multiply(h_inv, g)
        #update wight
        for i in range(len(w)):
            w[i] -= delta_w[i][0]
        #print w
        #update likelihood
        l_new = log_likelihood(x, y, w)
        delta_l = l - l_new
        if abs(delta_l) < 0.0000000001:
            return w
        l = l_new

def gradientDescent(x, y):
    w = [0.0, 0.0, 0.0]
    learning_rate = 1
    l = log_likelihood(x, y, w)
    while True:
        g = gradient(x, y, w)
        g = np.array(g).flatten()
        g = map(lambda a: a/len(x), g)
        for i in range(len(w)):
            w[i] -= (learning_rate*g[i])
        #print w
        l_new = log_likelihood(x, y, w)
        delta_l = l - l_new
        if abs(delta_l) < 0.0000000001:
            return w
        #learning_rate *= 0.95
        l = l_new

if __name__ == "__main__":
    [n, mx1, vx1, my1, vy1, mx2,vx2, my2, vy2] = map(lambda x: float(x), argv[1:])
    D1x = []
    D1y = []
    D2x = []
    D2y = []
    for i in range(int(n)):
        D1x.append([dataGenerator(mx1, vx1), dataGenerator(my1, vy1), 1.0])
        D1y.append(0.0)
        D2x.append([dataGenerator(mx2, vx2), dataGenerator(my2, vy2), 1.0])
        D2y.append(1.0)
    x = D1x + D2x
    y = D1y + D2y

    try:
        w = newtons_method(x, y)
    except:
        w = gradientDescent(x, y)

    #confusion matrix
    c = [[0, 0], [0, 0]]
    prob = sigmoid(x, w)
    for i in range(len(prob)):
        if y[i] == 1.0:
            if prob[i] >= 0.5:
                c[1][1] += 1
            else:
                c[1][0] += 1
        else:
            if prob[i] < 0.5:
                c[0][0] += 1
            else:
                c[0][1] += 1
    print np.array(c)
    if c[1][0]+c[0][1] == 0:
        specificity = 0.0
    else:
        specificity = float(c[0][1])/float(c[1][0]+c[0][1])
    if c[0][0]+c[1][1] == 0:
        sensitivity = 0.0
    else:
        sensitivity = float(c[0][0])/float(c[0][0]+c[1][1])
    acc = float(c[0][0]+c[1][1])/float(len(x))
    print "sensitivity =", sensitivity, "specificity =", specificity, "ACC =", acc
    print "Done."
    
    import matplotlib.pyplot as plt
    #plt.scatter([row[0] for row in D1x], [row[1] for row in D1x], color='red')
    #plt.scatter([row[0] for row in D2x], [row[1] for row in D2x], color='blue')
    red = []
    blue = []
    for i in range(len(prob)):
        if prob[i] >= 0.5:
            red.append(x[i])
        else:
            blue.append(x[i])
    plt.scatter([row[0] for row in red], [row[1] for row in red], color='red')
    plt.scatter([row[0] for row in blue], [row[1] for row in blue], color='blue')
    
    
    plt.show()
    
