from builtins import list
from collections import defaultdict
import csv

# episodes = defaultdict(list)
with open("/home/tdtrinh11/Documents/sentences.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter = ',')
    line_count = 0
    for row in reader:
        print(row)