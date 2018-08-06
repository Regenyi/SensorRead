''' thanks to @iamtrask and the stackoverflow community '''

import pprint
import numpy as np
import random
import data_processor


def create_empty_res_lists(num):
    global list_of_results
    global list_of_nns
    list_of_results = []
    list_of_nns = []
    for j in range(num):
        list_of_results.append([])
        list_of_nns.append([])


def gen_10_random_weight():
    return np.array([random.uniform(-1, 1) for _ in range(10)], np.float)


def create_random_nn():
    weights1 = gen_10_random_weight()
    weights2 = gen_10_random_weight()
    random_number = random.uniform(-1, 1)
    layer1_connecting_weights = [[random_number for i in range(2)] for j in range(10)]
    nn = [[weights1, weights2], layer1_connecting_weights]
    print("nnc  1", nn)
    return nn


def convert_to_nn(nn_part):
    weights1 = gen_10_random_weight()  # todo: második gentől ez honnan jön?
    weights2 = gen_10_random_weight()
    layer1_connecting_weights = nn_part[0][0]  # todo: [[[]]] 5d gonoszság bug lehetőség
    nn = [weights1, weights2, layer1_connecting_weights]
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
    list_of_results[iter].append(output)


def run_nn_on_input_data(iter, nn):
    if nn == 0:
        nn = create_random_nn()
    print("apnd 0", list_of_nns[0])
    list_of_nns[iter] = nn  # !!! elég-e a synapsisból ezt belerakni?
    print("apnd 1", list_of_nns[0])
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


def get_biggest_num(nn_output, index=0):
    nn_output_sorted = sorted(nn_output, key=lambda x: x[0], reverse=True)
    nn_index_and_its_biggest_num = (nn_output_sorted[0][0], index)
    return nn_index_and_its_biggest_num


def rank_nns(list_of_nn_outputs):
    nn_sorted_tuples = []
    pieces = len(list_of_results)
    for j in range(pieces):
        nn_sorted_tuples.append([])
    for i in range(pieces):
        nn_sorted_tuples[i].append(get_biggest_num(list_of_nn_outputs[i], i))
    nn_rank = sorted(nn_sorted_tuples, reverse=True)
    return nn_rank


def identify_nn(rank_tuple):
    return list_of_nns[rank_tuple[0][1]] # !!! jó outputot párositok-e jó nn-nel?


def breed(nn1, nn2):
    breeded = nn1  # this creates the proper dimensions for the new "empty" list
    for i in range(len(nn1)):
        for j in range(len(nn1[i])):
            for k in range(len(nn1[i][j])):
                breeded[i][j][k] = random_choose(nn1[i][j][k], nn2[i][j][k])
    return breeded


def breeder(nn_ranked_lists):
    print("     2", list_of_nns[0])
    next_gen_nn = []
    pieces = len(list_of_results)

    for j in range(pieces):
        next_gen_nn.append([])
    for i in range(10):
        nn_ranked_lists[i] = identify_nn(nn_ranked_lists[i])

    print("next 2", next_gen_nn[0])
    # keep 0 :
    next_gen_nn[0].append(nn_ranked_lists[0])
    print("next 3", next_gen_nn[0])
    print("rank 5", nn_ranked_lists[0])

    # breed 0 1 :
    next_gen_nn[1].append(breed(nn_ranked_lists[0], nn_ranked_lists[1]))
    # breed 0 2
    next_gen_nn[2].append(breed(nn_ranked_lists[0], nn_ranked_lists[2]))
    # breed 0 3
    next_gen_nn[3].append(breed(nn_ranked_lists[0], nn_ranked_lists[3]))
    # breed 0 4
    next_gen_nn[4].append(breed(nn_ranked_lists[0], nn_ranked_lists[4]))
    # breed 2 3
    next_gen_nn[5].append(breed(nn_ranked_lists[2], nn_ranked_lists[3]))
    # breed 2 4
    next_gen_nn[6].append(breed(nn_ranked_lists[2], nn_ranked_lists[4]))
    # breed 0 9
    next_gen_nn[7].append(breed(nn_ranked_lists[0], nn_ranked_lists[9]))
    # breed 8 9
    next_gen_nn[8].append(breed(nn_ranked_lists[8], nn_ranked_lists[9]))
    # new random nn
    new_rnd_nn = create_random_nn()
    next_gen_nn[9].append([new_rnd_nn[1]])
    print("     3", list_of_nns[0])
    print("next 9", next_gen_nn[0])
    return next_gen_nn


def print_res(index):  # input: page
    print("\n********** result list index is:", index, "***********\n")
    for row in range(len(processed_sensor_data)):
        x = str(list_of_results[index][row])
        print("input neurons: ", processed_sensor_data[row], '{1: >5} output: {0: >5}'.format(x.replace("\n", ''), " "))


def main():
    # *** INIT: *** #
    global processed_sensor_data
    np.random.seed(1)
    file = "data/save-1.tsv"
    processed_sensor_data = data_processor.read_lines(file, ' ')
    create_empty_res_lists(10)
    run_population(10)
    print("     1", list_of_nns[0])
    nextgen = breeder(rank_nns(list_of_results))

    # *** NEXT GEN LOOP STARTS FROM HERE: *** #
    # while exit_condition (0.99) not true or x = 100 loop
    x = 0
    while x < 3:
        create_empty_res_lists(10)
        for i in range(10):
            nn = convert_to_nn(nextgen[i])
            run_population(1, i, nn)
        nextgen = breeder(rank_nns(list_of_results))  # todo: check loop
        # print(get_biggest_num(list_of_results[0]))
        x += 1

    print("**** RESULTS: *****")
    print("     9", nextgen[0])
    # print(list_of_results[0])


if __name__ == '__main__':
    main()

