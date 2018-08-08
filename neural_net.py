''' thanks to @iamtrask and the stackoverflow community '''

import pprint
import numpy as np
import random
import data_processor
import logging


def create_empty_res_lists(num):
    global list_of_nn_outputs
    global list_of_nns
    list_of_nn_outputs = []
    list_of_nns = []
    for j in range(num):
        list_of_nn_outputs.append([])
        list_of_nns.append([])


def gen_random_weights():
    return np.array([random.uniform(-1, 1) for _ in range(10)], np.float)


def create_random_nn():
    weights1 = gen_random_weights()
    weights2 = gen_random_weights()
    random_number = random.uniform(-1, 1)
    layer1_connecting_weights = [[random_number for i in range(2)] for j in range(10)]
    nn = [[weights1, weights2], layer1_connecting_weights]
    # logging.debug("nnr  1", nn)
    logging.debug("nnr w1 {}".format(weights1))
    # logging.debug("nnr lc", layer1_connecting_weights)
    return nn


def random_choose(num1, num2):
    return random.choice([num1, num2])


def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def run_nn_on_line(line, nn, iter):
    # forward propagation:
    input = processed_sensor_data[line]
    layer1 = nonlin(np.dot((nn[1]), ([[input[0]], [input[1]]])))  # 2x10, 1x2
    output = nonlin(np.dot(([nn[0][0], nn[0][1]]), layer1))
    list_of_nn_outputs[iter].append(output)


def run_nn_on_input_data(iter, nn):
    if nn == 0:
        nn = create_random_nn()
    # print("apnd 0", list_of_nns[0])
    list_of_nns[iter] = nn
    # print("apnd 1", list_of_nns[0])
    line = 0
    while line != len(processed_sensor_data):
        run_nn_on_line(line, nn, iter)
        line += 1


def run_population(num, iter=0, nn=0):
    if iter == 0:
        for i in range(num):
            run_nn_on_input_data(i, nn)
    if iter != 0:
        for i in range(num):
            run_nn_on_input_data(iter, nn)


def get_biggest_num(nn_output):
    return np.amax(nn_output)


def rank_nns(lst_of_nn_outputs):
    biggest_num_unsorted_tuples = []
    pieces = len(list_of_nn_outputs)
    for j in range(pieces):
        biggest_num_unsorted_tuples.append([])
    for i in range(pieces):
        biggest_num_unsorted_tuples[i] = get_biggest_num(list_of_nn_outputs[i]), i
    biggest_num_and_source_nn_index = sorted(biggest_num_unsorted_tuples, reverse=True)  # pl: ((num1, 3) , (num2, 6))
    return biggest_num_and_source_nn_index


def identify_nn(biggest_num_and_source_nn_index):
    return list_of_nns[biggest_num_and_source_nn_index[1]]  # !!! jó outputot párositok-e jó nn-nel? úgy tűnik mintha jót


def breed(nn1, nn2):
    breeded = nn1  # this creates the proper dimensions for the new "empty" list
    for i in range(len(nn1)):
        for j in range(len(nn1[i])):
            for k in range(len(nn1[i][j])):
                breeded[i][j][k] = random_choose(nn1[i][j][k], nn2[i][j][k])
    return breeded


def breeder(biggest_num_and_source_nn_index_list):
    # print("     2", list_of_nns[0])
    next_gen_nn = []
    nn_ranked_lists = []
    pieces = len(list_of_nn_outputs)
    # print("1 lst_nns", list_of_nns)

    for j in range(pieces):
        next_gen_nn.append([])
        nn_ranked_lists.append([])
    for i in range(10):
        # print("bigest 0", biggest_num_and_source_nn_index_list[i])
        nn_ranked_lists[i] = identify_nn(biggest_num_and_source_nn_index_list[i])  # !!! felülirom? jónak tűnik

        # print("ranked_list", nn_ranked_lists[i])
        # print("biggest_nu2", biggest_num_and_source_nn_index_list[i])
        # x = biggest_num_and_source_nn_index_list[i][1]
        # print("list_of_nns", list_of_nns[x])
        # print("nns results", list_of_nn_outputs[x])

    # print("next 2", next_gen_nn[0])
    # keep 0 :
    next_gen_nn[0] = nn_ranked_lists[0]
    print("2 rank_lst", next_gen_nn[0])
    # print("3 nextgen0", next_gen_nn[0])
    # print("next 3", next_gen_nn[0])
    # print("rank 5", nn_ranked_lists[0])

    # breed 0 1 :
    next_gen_nn[1] = breed(nn_ranked_lists[0], nn_ranked_lists[1])
    # breed 0 2
    next_gen_nn[2] = breed(nn_ranked_lists[0], nn_ranked_lists[2])
    # breed 0 3
    next_gen_nn[3] = breed(nn_ranked_lists[0], nn_ranked_lists[3])
    # breed 0 4
    next_gen_nn[4] = breed(nn_ranked_lists[0], nn_ranked_lists[4])
    # breed 2 3
    next_gen_nn[5] = breed(nn_ranked_lists[2], nn_ranked_lists[3])
    # breed 2 4
    next_gen_nn[6] = breed(nn_ranked_lists[2], nn_ranked_lists[4])
    # breed 0 9
    next_gen_nn[7] = breed(nn_ranked_lists[0], nn_ranked_lists[9])
    # breed 8 9
    next_gen_nn[8] = breed(nn_ranked_lists[8], nn_ranked_lists[9])
    # new random nn
    new_rnd_nn = create_random_nn()
    next_gen_nn[9] = new_rnd_nn
    # print("     3", list_of_nns[0])
    # print("next 9", next_gen_nn[0])
    return next_gen_nn


def main():
    # *** INIT: *** #
    # logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    global processed_sensor_data
    np.random.seed(1)
    file = "data/save-1.tsv"
    processed_sensor_data = data_processor.read_lines(file, ' ')
    create_empty_res_lists(10)
    run_population(10)
    # print("     1", list_of_nns[0])
    biggest_num_and_source_nn_index_list = rank_nns(list_of_nn_outputs)
    nextgen = breeder(biggest_num_and_source_nn_index_list)
    print("5 rank_lst", nextgen[0])
    # print("bigest 1", get_biggest_num(list_of_nn_outputs))
    # print("_____4", nextgen[8])
    # print("_____5", nextgen[9])

    # *** NEXT GEN LOOP STARTS FROM HERE: *** #
    # while exit_condition (0.99) not true or x = 100 loop
    x = 0
    while x < 2:
        # print("while loop", x)
        create_empty_res_lists(10)

        for i in range(10):
            # print("for loop", i)
            nn = nextgen[i]
            run_population(1, i, nn)

        # print("4 lst_nns", list_of_nns)
        nextgen = breeder(rank_nns(list_of_nn_outputs))  # todo: check loop, seems ok
        # print("5 nextgen", nextgen[0])
        # print("bigest n", get_biggest_num(list_of_nn_outputs))
        x += 1

    print("**** RESULTS: *****")
    # print("     9", nextgen[0])
    # print(list_of_nn_outputs[0])


if __name__ == '__main__':
    main()

