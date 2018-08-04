import csv


def read_raw_data(file, delim):
    preprocessed_sensor_data = []
    with open(file) as data:
        reader = csv.reader(data, delimiter=delim)
        for row in reader:
            temp_string_cells = row[2:4]
            temp_row_line = []
            for cell in temp_string_cells:
                float_cell = float(cell.replace(',', '.'))
                if float_cell > -0.2:
                    if float_cell < 0.3:
                        float_cell = 0.0
                temp_row_line.append(float_cell)
            preprocessed_sensor_data.append(temp_row_line)
    return preprocessed_sensor_data


def process_data(preprocessed_sensor_data):
    filtered_data = []
    name = "kor"
    i = 1
    repeated_line = 0
    non_repeated_line = 0
    empty_line = [0.0, 0.0]
    for row in preprocessed_sensor_data:
        if row == empty_line:
            repeated_line += 1
        if row != empty_line:
            filtered_data.append(row)
            non_repeated_line += 1
            repeated_line = 0
        if non_repeated_line > 70 and repeated_line >= 5:
            save_to_file(filtered_data, name+('{0:0=2d}.{1}'.format(i, 'csv')))
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


def read_lines(file, delim):
    processed_sensor_data = list()
    with open(file) as data:
        reader = csv.reader(data, delimiter=delim)
        for row in reader:
            temp_row_line = []
            for cell in row:
                float_cell = float(cell)
                temp_row_line.append(float_cell)
            processed_sensor_data.append(temp_row_line)
        return processed_sensor_data


def main():
    file = 'data/kor.tsv'
    raw_data = read_raw_data(file, '\t')
    process_data(raw_data)


if __name__ == '__main__':
    main()
