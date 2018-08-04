''' thanks to @iamtrask and the stackoverflow community '''

import numpy as np
import random
import data_processor

file = "data/save-1.tsv"
processed_sensor_data = []
np.random.seed(1)


def create_empty_res_lists(num):
    global list_of_results
    list_of_results = []
    for j in range(num):
        list_of_results.append([])


def array_randomizer():
    return np.array([random.uniform(-1, 1) for _ in range(10)], np.float)


def create_random_nn():
    weights1 = array_randomizer()
    weights2 = array_randomizer()
    random_number = random.uniform(-1, 1)
    layer1_connecting_weights = [[random_number for i in range(2)] for j in range(10)]
    nn = [weights1, weights2, layer1_connecting_weights]
    return nn


def convert_to_nn(nn_part):
    weights1 = array_randomizer()  # todo: második gentől ez honnan jön?
    weights2 = array_randomizer()
    temp2d_list = []
    for i in range(len(nn_part[0])):
        x = np.array(nn_part[0][i]).flatten()
        temp2d_list.append(x)
    layer1_connecting_weights = temp2d_list[:10]  # todo: bug! ez most 10, közben meg 185
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


def random_choose(num1, num2):
    return random.choice([num1, num2])


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def run_nn_on_line(line, nn, iter):
    # forward propagation:
    input = processed_sensor_data[line]
    layer1 = nonlin(np.dot((nn[2]), ([[input[0]], [input[1]]])))  # 2x10, 1x2
    output = nonlin(np.dot(([nn[0], nn[1]]), layer1))
    list_of_results[iter].append(output)


def run_nn_on_input_data(iter, nn):
    # nn = create_random_nums()
    line = 0
    while line != len(processed_sensor_data):
        run_nn_on_line(line, nn, iter)
        line += 1


def run_population(num, nn=create_random_nn()):
    for i in range(num):
        run_nn_on_input_data(i, nn)


def get_biggest_num(nn, index=0):
    nn_sorted = sorted(nn, key=lambda x: x[0], reverse=True)
    nn_index_and_its_biggest_num = (nn_sorted[0][0], index)
    return nn_index_and_its_biggest_num


def rank_nns(list_of_nns):
    nn_sorted_tuples = []
    pieces = len(list_of_results)
    for j in range(pieces):
        nn_sorted_tuples.append([])
    for i in range(pieces):
        nn_sorted_tuples[i].append(get_biggest_num(list_of_nns[i], i))
    nn_rank = sorted(nn_sorted_tuples, reverse=True)
    return nn_rank


def identify_nn(rank_tuple):
    return list_of_results[rank_tuple[0][1]]


def breed(nn1, nn2):
    breeded = nn1  # this creates the proper dimensions for the new "empty" list
    for i in range(len(nn1)):
        for j in range(len(nn1[i])):
            for k in range(len(nn1[i][j])):
                breeded[i][j][k] = random_choose(nn1[i][j][k], nn2[i][j][k])
    return breeded


def breeder(nn_ranked_list):
    next_gen_nn = []
    pieces = len(list_of_results)
    for j in range(pieces):
        next_gen_nn.append([])
    for i in range(10):
        nn_ranked_list[i] = identify_nn(nn_ranked_list[i])

    # breed 0 1 :
    next_gen_nn[0].append(breed(nn_ranked_list[0], nn_ranked_list[1]))
    # breed 0 2
    next_gen_nn[1].append(breed(nn_ranked_list[0], nn_ranked_list[2]))
    # breed 0 3
    next_gen_nn[2].append(breed(nn_ranked_list[0], nn_ranked_list[3]))
    # breed 0 4
    next_gen_nn[3].append(breed(nn_ranked_list[0], nn_ranked_list[4]))
    # breed 2 3
    next_gen_nn[4].append(breed(nn_ranked_list[2], nn_ranked_list[3]))
    # breed 2 4
    next_gen_nn[5].append(breed(nn_ranked_list[2], nn_ranked_list[4]))
    # breed 0 9
    next_gen_nn[6].append(breed(nn_ranked_list[0], nn_ranked_list[9]))
    # breed 7 9
    next_gen_nn[7].append(breed(nn_ranked_list[7], nn_ranked_list[9]))
    # breed 8 9
    next_gen_nn[8].append(breed(nn_ranked_list[8], nn_ranked_list[9]))
    # new random nn
    new_rnd_nn = create_random_nn()
    next_gen_nn[9].append(new_rnd_nn[2])
    return next_gen_nn


def print_res(index):  # input: page
    print("\n********** result list index is:", index, "***********\n")
    for row in range(len(processed_sensor_data)):
        x = str(list_of_results[index][row])
        print("input neurons: ", processed_sensor_data[row], '{1: >5} output: {0: >5}'.format(x.replace("\n", ''), " "))


def tester():
    # gen_test_3darrays()
    # breed(x0, x1)
    # breed(z, z2)
    # print_res(2)
    return 0


def main():
    # *** INIT: *** #
    global processed_sensor_data
    processed_sensor_data = data_processor.read_lines(file, ' ')
    create_empty_res_lists(10)
    run_population(10)
    nextgen = breeder(rank_nns(list_of_results))

    # *** NEXT GEN LOOP STARTS FROM HERE: *** #
    # while exit_condition (0.99) not true or x = 100 loop

    nn = convert_to_nn(nextgen[0])  # todo: valójában itt a 10 db-t kell majd beadni egyesével (run population)
    create_empty_res_lists(10)
    run_population(10, nn)
    print("***********")
    print(list_of_results[1])
    nextgen = breeder(rank_nns(list_of_results))


if __name__ == '__main__':
    main()




