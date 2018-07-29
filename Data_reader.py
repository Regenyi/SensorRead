import csv
import datetime

filename = 'haromszog.tsv'
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
    if first_coord[0:3] == "0,2" or first_coord[0:3] == "0,1" or first_coord[0:3] == "0,0":
        filtered_data.append("string")
    else:
        filtered_data.append(row)



for row in filtered_data:
    print(row[0], ':: '.join(row[2:5]))


with open('new.csv', 'w') as csvout:
    csvout = csv.writer(csvout)
    for row in filtered_data:
        csvout.writerows(row[2:5])
