import sys
from sys import argv
import numpy as np

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

def LU_decomposition(M):
    n = len(M)
    L = [[0.0]*n for i in xrange(n)]
    U = [[0.0]*n for i in xrange(n)]

    for j in xrange(n):
        L[j][j] = 1
        for i in xrange(j+1):
            U[i][j] = M[i][j] - sum(U[k][j]*L[i][k] for k in xrange(i))
        for i in xrange(j, n):
            L[i][j] = (M[i][j] - sum(U[k][j]*L[i][k] for k in xrange(j))) / U[j][j]
    return (L, U)

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

def main():
    points = []
    if len(argv) < 4:
        print "You need to input 3 arguments: File, polynomial bases, lambda"
        sys.exit()
    try:
        with open(argv[1], 'r') as f:
            for line in f:
                points.append(line.split(','))
    except:
        print "File not found."
        sys.exit()
    bases = int(argv[2])
    l = float(argv[3])

    A = []
    b = []
    for p in points:
        x = float(p[0])
        y = float(p[1])
        row = []
        for i in range(bases):
            row.append(pow(x, i))
        A.append(list(reversed(row)))
        b.append(y)
    At = transpose(A)
    b = np.reshape(b, (len(b), 1))
    AtA = multiply(At, A)
    for i in range(len(AtA)):
        AtA[i][i] += l
    Atb = multiply(At, b)
    L, U = LU_decomposition(AtA)
    #x = multiply(multiply(inverse(U), inverse(L)), Atb)
    x = multiply(inverse(AtA), Atb)


    #calculate error
    Ax = multiply(A, x)
    Ax = np.reshape(Ax, (len(Ax, )))
    b = np.reshape(b, (len(b), ))
    s = 0
    for i in xrange(len(Ax)):
        s += abs(b[i] - Ax[i])

    print "coef =", np.array(x).reshape(len(x),)
    print "MAE =", s/len(Ax)

if __name__ == "__main__":
    main()
