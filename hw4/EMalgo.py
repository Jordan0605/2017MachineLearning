from keras.datasets import mnist
import numpy as np
from random import random
from math import log
from sys import float_info
from confusion_matrix import *

def gray2binary(data, threshold):
    for i in range(len(data)):
        data[i] = map(lambda x: 1 if x >= threshold else 0, data[i])
    return data

def log_likelihood(x, z, pi, mu):
    N = len(x)
    D = len(x[0])
    K = len(pi)
    
    mu_copy = list(mu)
    for k in range(K):
        for d in range(D):
            if mu_copy[k][d] < float_info.min:
                mu_copy[k][d] = float_info.min
            elif mu_copy[k][d] > 0.99999999999:
                mu_copy[k][d] = 0.99999999999

    l = 0.0
    for n in range(N):
        for k in range(K):
            try:
                partial_l = log(pi[k])
            except:
                partial_l = log(float_info.min)
            for i in range(D):
                partial_l += (x[n][i]*log(mu_copy[k][i]) + (1-x[n][i])*log(1-mu_copy[k][i]))
            l += (z[n][k]*partial_l)
    return l

def Bernoulli_Mixture_Model(x, K):
    #init
    N = len(x)
    D = len(x[0])
    K = K

    z = [[0.0 for i in range(K)] for j in range(N)]
    mu = [[random() for i in range(D)] for j in range(K)]
    pi = [1.0/float(K) for i in range(K)]
    
    iteration = 0
    delta = 1.0
    l = log_likelihood(x, z, pi, mu)
    while delta > 0.0000000001 and iteration < 10000:
        iteration += 1
        if iteration%100 == 0:
            print "iteration", iteration
        #Expectation step
        for n in range(N):
            for k in range(K):
                z[n][k] = 1.0
                for i in range(D/2):
                    z[n][k] *= ((mu[k][i]*10)**x[n][i] * ((1.0-mu[k][i])*10)**(1-x[n][i]))
                    j = i + D/2
                    z[n][k] *= (mu[k][j]**x[n][j] * (1.0-mu[k][j])**(1-x[n][j]))
                    #print "z", n, k, z[n][k]
                    """
                    if x[n][i] == 1:
                        z[n][k] *= (mu[k][i]*2)
                    elif x[n][i] == 0:
                        z[n][k] *= ((1-mu[k][i])*2)
                    else:
                        print "motherfucker."
                    """
                z[n][k] *= pi[k]
                devisor = 0.0
                for m in range(K):
                    partial_devisor = 1.0
                    for i in range(D/2):
                        partial_devisor *= ((mu[m][i]*10)**x[n][i] * ((1-mu[m][i])*10)**(1-x[n][i]))
                        j = i + D/2
                        partial_devisor *= (mu[m][j]**x[n][j] * (1-mu[m][j])**(1-x[n][j]))
                        #print "part", m, i, partial_devisor
                        """
                        if x[n][i] == 1:
                            partial_devisor *= (mu[m][i]*2)
                        elif x[n][i] == 0:
                            partial_devisor *= ((1-mu[m][i])*2)
                        """
                    devisor += (pi[m]*partial_devisor)
                if devisor == 0:
                    devisor = float_info.min
                z[n][k] /= devisor
        print np.array(z)
        #Maximization step
        N_m = [0.0 for i in range(K)] 
        
        """
        mu = np.dot(np.transpose(z), x)
        for m in range(K):
            mu[m] = np.array(mu[m]).dot(1.0/float(cluster_count[m]))
        """
        mu = [[0.0 for i in range(D)] for j in range(K)]
        pi = [0.0 for i in range(K)]
        """
        for m in range(K):
            for n in range(N):
                N_m[m] += z[n][m]
            pi[m] = float(N_m[m])/float(N)

        mu = np.dot(np.transpose(z), x)
        for m in range(K):
            mu[m] = np.array(mu[m]).dot(1.0/float(N_m[m]))
        """

        for k in range(K):
            for n in range(N):
                N_m[k] += z[n][k]
                for d in range(D):
                    mu[k][d] += (z[n][k]*x[n][d])
            for d in range(D):
                mu[k][d] /= N_m[k]
            pi[k] = float(N_m[k])/float(N)
        
        l_new = log_likelihood(x, z, pi, mu)
        delta = abs(l_new - l)
        l = l_new
        print np.array(mu)
        print np.array(pi)
    print "total iteration =", iteration
    return z

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train.reshape(len(x_train), -1)[:100]
    y = y_train[:100]

    num_pixels = len(x[0])
    m = np.mean(x)
    x = gray2binary(x, m)
    print "finish gray2binary."

    #BMM
    z = Bernoulli_Mixture_Model(x, 10)
    predict = []
    for i in range(len(z)):
        predict.append(np.argmax(z[i]))
    
    print confusion_matrix(y, predict)
