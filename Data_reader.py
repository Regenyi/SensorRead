import csv
import datetime

filename = 'test01.tsv'
with open(filename) as ds:
    reader = csv.reader(ds, delimiter='\t')
    data = [r for r in reader]
    # data = []
    # for r in row:
    #     data.append(r)

filtered_data = []
for row in data:
    #  print(datetime.datetime.utcfromtimestamp(int(row[0])).strftime('%Y-%m-%d %H:%M:%S'))
    first_coord = row[2]

    if first_coord[0] == "0":  # or first_coord[0:3] == "-0,0":
        filtered_data.append("string")
    else:
        filtered_data.append(row)



for row in filtered_data:
    print(':: '.join(row[2:5]))



