from collections import defaultdict
import csv

episodes = defaultdict(list)
with open("./sentences.csv", "r") as sentences_file:
    reader = csv.reader(sentences_file, delimiter = ',')

    for row in reader:
        episodes[row[1]].append(row[4])

for episodes_id, text in episodes.items():
    episodes[episodes_id] = "".join(text)
# print(episodes["1"])

corpus = []
for id, episodes in sorted(episodes.items(), key=lambda t: int(t[0])):
    corpus.append(episodes)

print(corpus[0])