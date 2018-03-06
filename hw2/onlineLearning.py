from sys import argv

def factorial(x):
    result = 1
    for i in range(1, x+1):
        result *= i
    return result

def gamma(x):
    return factorial(x-1)

def beta_prior(p, a, b):
    return pow(p, a-1)*pow(1-p, b-1)*(gamma(a+b)/(gamma(a)*gamma(b)))

def binomial_likelihood(p, n, m):
    return (factorial(n)/(factorial(m)*factorial(n-m)))*pow(p, m)*pow(1-p, n-m)

if __name__ == "__main__":
    a = int(argv[2])
    b = int(argv[3])
    with open(argv[1], 'r') as f:
        for line in f:
            line = line.strip()
            print line
            arr = map(lambda x: int(x) ,list(line))
            m = 0
            for i in arr:
                m += i
            p = float(m)/float(len(arr))
            print "MLE(p) =", p, "m =", m 
            n = len(arr)
            p = binomial_likelihood(p, n, m)
            print "Binomial likelihood =", p, "Beta prior =", beta_prior(p, a, b), "posterior probability =", p*beta_prior(p, a, b) 
