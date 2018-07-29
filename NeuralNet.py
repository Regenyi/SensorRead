import csv

file = "data/save-1.tsv"


def reader():
    with open(file) as dat:
        reader = csv.reader(dat, delimiter=' ')
        data = []
        for r in reader:
            yield [float(i) for i in r]
            data.append(r)

    for row in data:
        print(row)


dat = []


def readLines():
    def conv(s):
        try:
            s=float(s)
        except ValueError:
            pass
        return s

    with open(file) as data:
        reader = csv.reader(data)
        for row in reader:
            for cell in row:
                y = conv(cell)
                dat.append(y)


readLines()

for row in dat:
    print(row)

#


