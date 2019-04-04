import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

corpus = """
 Google and Facebook are strangling the free press to death. Democracy is the loserGoogle an 
Your 60-second guide to security stuff Google touted today at Next '18
A Guide to Using Android Without Selling Your Soul to Google
Review: Lenovo’s Google Smart Display is pretty and intelligent
Google Maps user spots mysterious object submerged off the coast of Greece - and no-one knows what it is
Android is better than IOS
In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency
is a numerical statistic that is intended to reflect
how important a word is to a document in a collection or corpus.
It is often used as a weighting factor in searches of information retrieval
text mining, and user modeling. The tf-idf value increases proportionally
to the number of times a word appears in the document
and is offset by the frequency of the word in the corpus
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

