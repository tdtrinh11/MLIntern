import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

corpus = """
One of the main advantages of using Google as an advertising platform is its immense reach. Google handles more than 40,000 search queries every second, a total of more than 1.2 trillion web searches every single year. As Google becomes increasingly sophisticated – in part to its growing reliance on its proprietary artificial intelligence and machine learning technology, RankBrain – this amazing search volume is likely to increase, along with the potential for advertisers to reach new customers.
Put simply, no other search engine can offer the potential audience that Google can. This vast potential source of prospective customers alone makes Google an excellent addition to your digital marketing strategy, but when combined with Google’s increasingly accurate search results, it’s easy to see why AdWords is the most popular and widely used PPC platform in the world.
As the world’s most popular and widely used search engine, Google is considered the de facto leader in online advertising. Fielding more than 3.5 billion search queries every single day, Google offers advertisers access to an unprecedented and unequaled potential audience of users who are actively looking for goods and services.
""".split("\n")[1:-1]
# print(corpus)
# clearing and tokenizing
l_A = corpus[0].lower().split()
l_B = corpus[1].lower().split()
l_C = corpus[2].lower().split()

# Calculating bag of words
#@@ tao ra chuoi cac tu khong trung nhau co trong doan van
word_set = set(l_A).union(set(l_B)).union(set(l_C))

# @@ tao tu dien
word_dict_A = dict.fromkeys(word_set, 0)
word_dict_B = dict.fromkeys(word_set, 0)
word_dict_C = dict.fromkeys(word_set, 0)

# @@ tinh so lan xuat hien cua cac tu co trong word_set trong tung doan
for word in l_A:
    word_dict_A[word] += 1
for word in l_B:
    word_dict_B[word] += 1
for word in l_C:
    word_dict_C[word] += 1

# @@ tao tu dien luu tan xuat xuat hien cua tu trong cau
def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count / sum_nk
    return tf

tf_A = compute_tf(word_dict_A, l_A)
tf_B = compute_tf(word_dict_B, l_B)
tf_C = compute_tf(word_dict_C, l_C)

# @@
def compute_idf(strings_list):
    n = len(strings_list) # @@ tong so chuoi trong strings_list
    idf = dict.fromkeys(strings_list[0].keys(), 0)  # @@ tu dien cac tu, so lan xuat hien
    # @@ dem so chuoi ma tu trong tu dien idf xuat hien
    for l in strings_list:
        for word, count in l.items():
            if count > 0:
                idf[word] += 1

    # @@ tinh idf theo cong thuc
    for word, v in idf.items():
        idf[word] = math.log(n / float(v))
    return idf

idf = compute_idf([word_dict_A, word_dict_B, word_dict_C])

def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf

tf_idf_A = compute_tf_idf(tf_A, idf)
tf_idf_B = compute_tf_idf(tf_B, idf)
tf_idf_C = compute_tf_idf(tf_C, idf)

print(len(tf_idf_A))
# print(len(tf_idf_B))
# print(tf_idf_C)