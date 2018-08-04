import csv

sensor_data = []
file = 'data/haromszog.tsv'


def read_raw_data(delimeter):
    with open(file) as data:
        reader = csv.reader(data, delimiter=delimeter)
        for row in reader:
            temp_string_cells = row[2:4]
            temp_row_line = []
            for cell in temp_string_cells:
                float_cell = float(cell.replace(',', '.'))
                if float_cell > -0.2:
                    if float_cell < 0.3:
                        float_cell = 0.0
                temp_row_line.append(float_cell)
            sensor_data.append(temp_row_line)


def process_data():
    filtered_data = []
    name = "new"
    i = 0
    repeated_line = 0
    non_repeated_line = 0
    empty_line = [0.0, 0.0]
    for row in sensor_data:
        if row == empty_line:
            repeated_line += 1
            # jump to next line
        if row != empty_line:
            filtered_data.append(row)
            non_repeated_line += 1
            repeated_line = 0
        if non_repeated_line > 70 and repeated_line >= 5:
            save_to_file(filtered_data, name+str(i))
            filtered_data = []
            i += 1
            non_repeated_line = 0
            repeated_line = 0


def save_to_file(data, filename):
    with open(filename, 'w') as csvout:
        csvout = csv.writer(csvout)
        for row in data:
            csvout.writerow(row)


def print_data(data):
    for row in data:
        print(row)


def main():
    read_raw_data('\t')
    process_data()
    # print_data(sensor_data)


if __name__ == '__main__':
    main()
