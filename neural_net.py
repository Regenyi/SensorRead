# thanks to @iamtrask and the stackoverflow community

import csv
import numpy as np
import random


file = "data/save-1.tsv"
sensor_data = []


def create_empty_res_lists(num):
    global list_of_results
    list_of_results = []
    for j in range(num):
        list_of_results.append([])


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


def gen_test_3darrays():
    global x0, x1, z, z2
    x0 = np.zeros((3, 3, 3))
    x1 = np.ones((3, 3, 3))
    x = np.zeros((3, 2))
    y = np.zeros((2, 3))
    z = np.array([x, y])
    x = np.ones((3, 2))
    y = np.ones((2, 3))
    z2 = np.array([x, y])


def breed(neuralnet1, neuralnet2):
    breed_list = neuralnet1
    x0 = neuralnet1
    x1 = neuralnet2
    # print("nn1", x0)
    # print("nn2", x1)
    for i in range(len(x0)):
        for j in range(len(x0[i])):
            for k in range(len(x0[i][j])):
                breed_list[i][j][k] = random_choose(x0[i][j][k], x1[i][j][k])
    # print("res", breed_list)
    return breed_list


def random_choose(num1, num2):
    return random.choice([num1, num2])


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def run_nn_on_line(line, nn, iteration):
    # forward propagation:
    input = sensor_data[line]
    layer1 = nonlin(np.dot(np.matrix(nn[2]), np.matrix([[input[0]], [input[1]]])))  # 2x10, 1x2
    output = nonlin(np.dot(np.matrix([nn[0], nn[1]]), np.matrix(layer1)))
    list_of_results[iteration].append(output)


def run_nn_on_input_data(iteration):
    nn = create_random_nums()
    x = 0
    line = 0
    while x != len(sensor_data):
        run_nn_on_line(line, nn, iteration)
        line += 1
        x += 1


def run_population(num):
    for i in range(num):
        run_nn_on_input_data(i)


def print_res(index):  # input: page
    print("\n********** result list index is:", index, "***********\n")
    for row in range(len(sensor_data)):
        x = str(list_of_results[index][row])
        print("input neurons: ", sensor_data[row], '{1: >5} output: {0: >5}'.format(x.replace("\n", ''), " "))


read_lines()
create_empty_res_lists(10)
run_population(10)
# gen_test_3darrays()
# breed(x0, x1)
# breed(z, z2)
print_res(2)


# todo:
# 10 db listába mentem --
# a legmagasabb első oszlop száma alapján a 10 listából megtartom aki a legmagasabb volt.
# és breedeltetem a képlet alapján
# [ return (sorted^.to (!!0)._1)
# , foo 0 1
# , foo 0 2
# , foo 0 3
# , foo 0 4
# , foo 2 3
# , foo 2 4
# , foo 0 9
# , foo 8 9
# , newRandomNet]
# ez lesz a második generáció
# exit condition vagy 0.99-1 vagy 100 generáció










