from math import sqrt, log
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
