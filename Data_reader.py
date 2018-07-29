import csv
import datetime

filename = 'test01.csv'
with open(filename) as ds:
    reader = csv.reader(ds)
    data = [r for r in reader]
    # data = []
    # for r in row:
    #     data.append(r)

for row in data:
    print(datetime.datetime.utcfromtimestamp(int(row[0])).strftime('%H:%M:%S'))
    print(', '.join(row[2:4]))
    # print(row)


