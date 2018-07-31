# thanks to @iamtrask and the stackoverflow community

import csv
import numpy as np
import random
# from numpy import random

file = "data/save-1.tsv"
sensor_data = []
result = []


def read_lines():
    with open(file) as data:
        reader = csv.reader(data, delimiter=' ')
        for row in reader:
            temp_row_line = []
            for cell in row:
                float_cell = float(cell)
                temp_row_line.append(float_cell)
            sensor_data.append(temp_row_line)


def array_randomizer():
    return np.array([random.uniform(-1, 1) for _ in range(10)], np.float)


def create_random_nums():
    weights1 = array_randomizer()
    weights2 = array_randomizer()
    random_number = random.uniform(-1, 1)
    layer1_connecting_weights = [[random_number for i in range(2)] for j in range(10)]
    nn = [weights1, weights2, layer1_connecting_weights]
    return nn


def breed(neuralnet1, neuralnet2):
    breed_list = []
    for i in range(len(neuralnet1)):
        breed0 = random_choose(neuralnet1[i][0], neuralnet2[i][0])
        breed1 = random_choose(neuralnet1[i][1], neuralnet2[i][1])
        breed_list.append(breed0)
        breed_list.append(breed1)
        breed_list.append("\n")
    # print(breed_list)
    return breed_list


def random_choose(num1, num2):
    return random.choice([num1, num2])


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def run_neural_net(line, nn):
    # forward propagation:
    input = sensor_data[line]
    layer1 = nonlin(np.dot(np.matrix(nn[2]), np.matrix([[input[0]], [input[1]]])))  # 2x10, 1x2
    output = nonlin(np.dot(np.matrix([nn[0], nn[1]]), np.matrix(layer1)))
    result.append(output)


def loop():
    nn = create_random_nums()
    x = 0
    line = 0
    while x != len(sensor_data):
        run_neural_net(line, nn)
        line += 1
        x += 1


def print_res():
    for row in range(len(sensor_data)):
        x = str(result[row])
        print("input neurons: ", sensor_data[row], '{1: >5} output: {0: >5}'.format(x.replace("\n", ''), " "))


read_lines()
loop()
breed(result, sensor_data)
# print_res()









