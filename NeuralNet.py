import csv
import numpy as np
import random

file = "data/save-1.tsv"
sensor_data = []
result = []


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


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


def run_neuron_net(line, nn):
    # forward propagation

    input = sensor_data[line]
    layer1 = nonlin(np.dot(np.matrix(nn[2]), np.matrix([[input[0]], [input[1]]])))  # 2x10,  1x2
    output = nonlin(np.dot(np.matrix([nn[0], nn[1]]), np.matrix(layer1)))
    result.append(output)


def print_res():
    for row in range(len(sensor_data)):
        print(sensor_data[row], result[row])


def create_random_nums():
    rnd1 = array_randomizer()
    rnd2 = array_randomizer()
    random_number = random.uniform(-1, 1)
    layer1_con_weights = [[random_number for i in range(2)] for j in range(10)]
    nn = [rnd1, rnd2, layer1_con_weights]
    return nn


def loop():
    nn = create_random_nums()
    x = 0
    line = 0
    while x != len(sensor_data):
        run_neuron_net(line, nn)
        # nn3 = breed(nn1, nn2)
        line += 1
        x += 1


read_lines()
loop()
print_res()









