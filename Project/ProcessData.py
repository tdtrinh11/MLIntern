import os
from sklearn.datasets import load_files
from demo import FilterData

# from sklearn.datasets import load_mlcomp
# ss = """( ) [ ] { } ' " , . ! ? : ;""".split(' ')

twenty_train = load_files("../Data/20news-bydate/20news-bydate-train", 
                           description= None, categories=None, load_content = True, 
                           encoding='latin1', decode_error='strict', shuffle=True, 
                           random_state=42)

file_names = twenty_train.filenames         # filenames list > train.filenames
target_names = twenty_train.target_names    # groupnames list > train.target-names
target = twenty_train.target                # label list > train.label
data_train = twenty_train.data              # data list > train.data

# # print(file_names)
# # file chua ten cai file theo thu tu
# f = open('/home/tdtrinh11/data/train.filenames', 'w')
# for names in file_names:
#     line = names + '\n'
#     f.write(line)
# f.close()

# # print(target_names)
# # file chua ten nhom va id cua cac nhom
# f = open('/home/tdtrinh11/data/train.target-names', 'w')
# # f.write("groupName groupId'\n'")
# for i in range(len(target_names)):
#     line = target_names[i] + ' ' + str(i + 1) + '\n'
#     # print(line)
#     f.write(line)
# f.close()

# # print(len(target))
# # file chua label
# f = open('/home/tdtrinh11/data/train.label', 'w')
# for i in range(len(target)):
#     line = str(target[i] + 1) + '\n'
#     f.write(line)
# f.close()

# print(data_train[0].replace('\n', ' ').split(' '))
# def removeSystax(word):
#     cs = []
#     for i in range(len(word)):
#         if word[i] in ss:
#             # print(w[i])
#             cs.append(word[i])
#     for c in cs:
#         word = word.replace(c, '')
#     return word

