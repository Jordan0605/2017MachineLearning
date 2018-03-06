from sys import argv
from math import log, sqrt
from random import random

spare = 0.0
isSpareReady = False

def dataGenerator(m, std):
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

if __name__ == "__main__":
    m = float(argv[1])
    std = float(argv[2])
    data = []
    while True:
        data.append(dataGenerator(m, std))
        if len(data) == 1:
            estimateMean = data[0]
            estimateStd = 0.0
            print data[0], estimateMean, estimateStd
            continue
        #tmp_mean = round(reduce(lambda x, y: x+y, data)/len(data), 4)
        tmp_mean = estimateMean + (data[-1] - estimateMean)/len(data)
        tmp_mean = round(tmp_mean, 4)
        tmp_std = 0.0
        for i in data:
            tmp_std += (i-tmp_mean)**2
        tmp_std /= (len(data)-1)
        tmp_std = round(sqrt(tmp_std), 4)
        #tmp_std = (estimateStd*(len(data)-1) + (data[-1] - estimateMean)*(data[-1] - tmp_mean)) / len(data)
        #tmp_std = sqrt(tmp_std)
        #tmp_std = round(tmp_std, 4)
        print data[-1], tmp_mean, tmp_std
        if tmp_mean == estimateMean and tmp_std == estimateStd:
            print "Done."
            break
        estimateMean = tmp_mean
        estimateStd = tmp_std
