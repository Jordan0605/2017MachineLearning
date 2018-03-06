from sys import argv
from math import log, sqrt
from random import random, uniform
import numpy as np

spare = 0.0
isSpareReady = False

def dataGenerator(std):
    m = 0
    global isSpareReady
    global spare
    if isSpareReady:
        isSpareReady = False
        return round(spare*std + m, 4)
    s = 0.0
    while s >= 1 or s == 0.0:
        u = random()*2 - 1
        v = random()*2 - 1
        s = u**2 + v**2
    mu = sqrt(-2.0 * log(s) / s)
    spare = v * mu
    isSpareReady = True
    return round(std*u*mu + m, 4)

def linear_model(basis, std, w):
    x = uniform(-10, 10)
    y = 0.0
    for i in range(basis):
        y += (w[i]*(x**i))
    return x, y + dataGenerator(std)

def transpose(M):
    n = len(M)
    m = len(M[0])
    t = [[0]*n for i in xrange(m)]
    for i in range(m):
        for j in range(n):
            t[i][j] = M[j][i]
    return t

def multiply(p, q):
    a = [[0]*len(q[0]) for i in xrange(len(p))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            for k in range(len(q)):
                a[i][j] += p[i][k] * q[k][j]
    return a            

def partial_matrix(M, i, j):
    return [row[:j]+row[j+1:] for row in (M[:i]+M[i+1:])]

def determinant(M):
    if len(M) == 2:
        return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    d = 0
    for i in xrange(len(M)):
        d += ((-1)**i)*M[0][i]*determinant(partial_matrix(M,0,i))
    return d

def inverse(M):
    det = determinant(M)
    if len(M) == 2:
        return [[M[1][1]/det, -1*M[0][1]/det], [-1*M[1][0]/det, M[0][0]/det]]
    cofactor = []
    for i in xrange(len(M)):
        row = []
        for j in xrange(len(M)):
            part = partial_matrix(M, i, j)
            row.append(((-1)**(i+j))*determinant(part))
        cofactor.append(row)
    cofactor = transpose(cofactor)
    for i in xrange(len(cofactor)):
        for j in xrange(len(cofactor)):
            cofactor[i][j] = cofactor[i][j] / det
    return cofactor


if __name__ == "__main__":
    precision = float(argv[1])
    basis = int(argv[2])
    std = float(argv[3])
    w = []
    with open(argv[4], 'r') as f:
        for line in f:
            w.append(float(line))
    data = []
    old_coef = []
    
    while True:
        (x, y) = linear_model(basis, std, w)
        data.append((x, y))
        A = []
        b = []
        for p in data:
            row = []
            for i in range(basis):
                row.append(p[0]**i)
            A.append(list(reversed(row)))
            b.append(p[1])
        At = transpose(A)
        b = np.reshape(b, (len(b), 1))
        AtA = multiply(At, A)
        for i in range(len(AtA)):
            AtA[i][i] += (precision/std)
        Atb = multiply(At, b)
        new_coef = multiply(inverse(AtA), Atb)
        for i in range(len(new_coef)):
            new_coef[i] = map(lambda x: round(x, 6), new_coef[i])
        if len(data) == 1:
            old_coef = new_coef
            continue
        
        X = []
        for i in range(basis):
            X.append([x**(basis-1-i)])
        l = [[0.0 for i in range(basis)] for j in range(basis)]
        for i in range(basis):
            l[i][i] = precision
        mean = transpose(multiply(transpose(new_coef), X))[0][0]
        s = 1/std + multiply(multiply(transpose(X), l), X)[0][0]

        print "(%f, %f)" %(x, y), new_coef, mean, s
        if new_coef == old_coef:
            print "Done."
            break
        old_coef = new_coef
