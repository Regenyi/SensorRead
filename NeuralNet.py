import csv
import numpy as np
import random

file = "data/save-1.tsv"
dat = []
filter3 = []

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def reader():
    with open(file) as dat:
        reader = csv.reader(dat, delimiter=' ')
        data = []
        for r in reader:
            yield [float(i) for i in r]
            data.append(r)

    for row in data:
        print(row)


def readLines():
    def conv(s):
        try:
            s=float(s)
        except ValueError:
            pass
        return s

    with open(file) as data:
        reader = csv.reader(data)
        for row in reader:
            for cell in row:
                y = conv(cell)
                dat.append(y)


def training_start():
    np.random.seed(1)
    array1 = [random.uniform(-1, 1) for _ in range(10)]
    array2 = [random.uniform(-1, 1) for _ in range(10)]
    array3 = [random.uniform(-1, 1) for _ in range(10)]
    array4 = [random.uniform(-1, 1) for _ in range(10)]
    filter = []
    filter.append(array1)
    filter.append(array2)
    filter2 = []
    filter2.append(array3)
    filter2.append(array4)
    filter3.append(filter)
    filter3.append(filter2)


def get_matrix():
    for iter in xrange(1):
        # forward propagation
        l0 = filter3
        l1 = nonlin(np.dot(l0, ))


readLines()
training_start()
get_matrix()

# for row in dat:
#     print(row)





