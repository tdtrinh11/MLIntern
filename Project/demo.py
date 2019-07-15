import re

# w = '11 i\'m    trinh_tien-dat, trinh@ tien dat a an the'
# for i in range(len(w)):
#     # w[i] = re.sub('[^(\w | \' | \-)]', '', w[i])
#     w[i] = re.sub('[^A-Za-z]', ' ', w[i])
#     print(w[i])

def FilterData(para):
    f = open('/home/tdtrinh11/data/stopwords.english', 'r')
    stopwords_list = f.read().split('\n')
    f.close()

    list_of_words = re.sub('[^A-Za-z]', ' ', para).split()
    list_of_words = [w for w in list_of_words if not w in stopwords_list]
    return list_of_words

# w = FilterData(w)
# print(w)

# w = re.sub('[^A-Za-z]', ' ', w)
# list_w = w.split()
# print(list_w)
# list_w = [w for w in list_w if not w in stopwords_list]
# # print(w)
# print(list_w)