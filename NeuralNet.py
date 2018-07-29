import csv
import numpy as np
import random

file = "data/save-1.tsv"
dat = []
array1 = np.array([random.uniform(-1, 1) for _ in range(10)], np.float)


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def readLines():
    def conv(s):
        try:
            s=float(s)
        except ValueError:
            pass
        return s

    with open(file) as data:
        reader = csv.reader(data, delimiter=' ')
        for row in reader:
            temp = []
            for cell in row:
                y = float(cell)
                temp.append(y)
            dat.append(temp)


def training_start():
    array1 = np.array([random.uniform(-1, 1) for _ in range(10)])
    array2 = np.array([random.uniform(-1, 1) for _ in range(10)])
    array3 = np.array([random.uniform(-1, 1) for _ in range(10)])
    array4 = np.array([random.uniform(-1, 1) for _ in range(10)])



def get_matrix():
    for iter in range(1):
        # forward propagation
        l0 = dat[0]
        print(np.dot(np.matrix([[2], [5]]), np.matrix([[3, 2]])))
        print(np.dot(np.matrix([[3, 2]]), np.matrix([[2], [5]])))
        l1 = np.dot(np.matrix([[1], [5]]), np.matrix(array1))
        # l1 = nonlin(np.dot(l0, array1))
    # print(l1, array1)
    print(l1)


readLines()
training_start()
get_matrix()

# for row in dat:
#     print(row)





