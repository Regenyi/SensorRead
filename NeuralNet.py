import csv
import numpy as np
import random


file = "data/save-1.tsv"
dat = []
array1 = np.array([random.uniform(-1, 1) for _ in range(10)], np.float)
array2 = np.array([random.uniform(-1, 1) for _ in range(10)], np.float)
layer1_con_weights = [[1.0, 2.0],
                      [3.0,  4.0],
                      [5.0,  6.0],
                      [7.0,  8.0],
                      [9.0, 10.0],
                      [11.0, 12.0],
                      [13.0, 14.0],
                      [15.0, 16.0],
                      [17.0, 18.0],
                      [19.0, 20.0]]


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def readLines():
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
        # print("2x1" , np.dot(np.matrix([[2], [5], [1]]), np.matrix([[3, 2]])))
        # l1 = np.dot(np.matrix([[1], [5]]), np.matrix(array1))

        l1 = np.dot(np.matrix([[1, 5]]), np.matrix([array1, array2]))  # 10x2,  2x1
        l1 = nonlin(np.dot(np.matrix(layer1_con_weights), np.matrix([[1], [4]])))  # 2x10,  1x2
        output = nonlin(np.dot(np.matrix([array1, array2]), np.matrix(l1)))

        # l1 = nonlin(np.dot(l0, array1))
    print(output)

# 2 inputhoz a 2 output szám elmentése 2d-s listába

def print_res():
    for row in dat:
        print(row)


readLines()
training_start()
get_matrix()
# print_res()








