import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

corpus = """
Until recently, many advertisers viewed Google AdWords and Facebook Ads in an adversarial way. The two companies’ long-standing rivalry, often dramatized by technology media outlets, was taken as irrefutable evidence that the two platforms were in direct competition with one another, and that it was necessary for businesses of all sizes to make a difficult decision about which platform was right for their needs; a false dichotomy that remains confusing and misleading to those new to online advertising.
One of the main advantages of using Google as an advertising platform is its immense reach. Google handles more than 40,000 search queries every second, a total of more than 1.2 trillion web searches every single year. As Google becomes increasingly sophisticated – in part to its growing reliance on its proprietary artificial intelligence and machine learning technology, RankBrain – this amazing search volume is likely to increase, along with the potential for advertisers to reach new customers.
Put simply, no other search engine can offer the potential audience that Google can. This vast potential source of prospective customers alone makes Google an excellent addition to your digital marketing strategy, but when combined with Google’s increasingly accurate search results, it’s easy to see why AdWords is the most popular and widely used PPC platform in the world.
Although the two platforms are often positioned as competitors, nothing could be further from the truth in a practical sense. Many businesses are leveraging the strengths of advertising on Google and Facebook Ads in concert to achieve maximum visibility, increase leads and sales, and find new customers, adopting different strategies that align with the functionality of each platform and seeing remarkable return on their advertising spend.
In this guide, we’ll examine what sets Google AdWords and Facebook Ads apart, how the two ad platforms work, and why you should consider using both as part of your wider digital marketing strategy.
As the world’s most popular and widely used search engine, Google is considered the de facto leader in online advertising. Fielding more than 3.5 billion search queries every single day, Google offers advertisers access to an unprecedented and unequaled potential audience of users who are actively looking for goods and services.
Paid search focuses on the targeting of keywords and the use of text-based advertisements. Advertisers using AdWords bid on keywords – specific words and phrases included in search queries entered by Google users – in the hopes that their ads will be displayed alongside search results for these queries. Each time a user clicks on an ad, the advertiser is charged a certain amount of money, hence the name “pay-per-click advertising.” PPC bidding and bid optimization is a complex topic, and beyond the scope of this guide, but essentially, users are paying for the potential to find new customers based on the keywords and search terms they enter into Google.
Facebook Ads is a prime example of what is known as “paid social,” or the practice of advertising on social networks. With the highest number of monthly active users (or MAUs) of any social network in the world, Facebook has become a highly competitive and potentially lucrative element of many business’ digital advertising strategies.
When AdWords first launched in 2000 (with a grand total of just 350 advertisers), the text-based ads that Google served alongside its search results were rudimentary, to say the least – but they did contain many of the same elements that can be seen in today’s ads.
Similarly to Google AdWords, Facebook boasts a truly vast global audience. With more than 1.55 BILLION monthly active users – more than one-fifth of the entire world’s population, and that’s not counting inactive or infrequently used accounts – Facebook has no rival when it comes to the enormity of its audience. However, rather than exposing advertisers and their messaging to this vast audience, the true strength of Facebook’s immense audience lies in the potential granularity with which advertisers can target Facebook’s users.
Unlike their comparatively dry, text-based PPC cousins, Facebook ads are powerfully visual. The very best Facebook ads blend in seamlessly with the videos, images, and other visual content in users’ News Feeds, and this enables advertisers to leverage not only the strongly persuasive qualities of visual ads but to do so in a way that conveys the aspirational messaging that makes high-quality ads so compelling.
Businesses and marketers experimenting with Facebook Ads are often impressed by the granularity of its targeting options, as well as the tools they have at their disposal for creating beautiful, engaging ads. However, one element of Facebook Ads that consistently takes newcomers by surprise is the potential return on investment that advertising on Facebook offers, and how far savvy advertisers can stretch a limited ad budget on the platform.
To learn more about how to maximize the impact of your Google AdWords and Facebook Ads campaigns, check out the free lessons at WordStream’s PPC University. Divided into three distinct tracks beginner, intermediate, and advanced users, PPC U has everything you need to master paid search and paid social, and to help even the most modest advertising budget work harder and smarter.
The tough war between Google vs Facebook (the two giants in technology space) began ever since Facebook started gaining more popularity than any other brand in the social web. Google’s product portfolio consists of more than 100 products, as Google wanted to have its presence in every product which is used on Internet. This led Google into concentrating on continuous acquisitions in the market mainly Orkut, Youtube and others.
Google Started off late in 1998 by three people Larry page, Sergey Brin, Eric Schmidt. The company started with a core idea of being one-stop search engine for Internet users in getting all the information around the world in a single mouse click. Google’s strategy is to constantly innovate and engage in acquisition and partnering with other companies beyond the core business of providing search engine. Google’s broad classification of products includes web-based products, operating systems, desktop and mobile applications, hardware etc.
There is a saying “Curiosity is the mother of invention”, but here the idea of search “curiosity” itself made a great revolution in web space. Google’s revenue generation model is mainly from advertising and its main motive is to make people stay more time on its webpage. Along with search engine advertising, there are products like google adwords and google adsense which allows marketers to advertise and earn revenue respectively. Adwords and adsense contribute heavily to the revenue generated by Google. Google advertising covers 41% of the $31 billion US advertising market. When Google has shown such a rapid growth ever since its launch, what made Larry Page to go for Google plus? Which was a similar concept of like and share in Facebook!
The revenue from search advertising were slowly declining as the companies started concentrating on advertising in social networking giant ruling the social web, Facebook. Facebook has a user base of over 800 million and a survey says that an FB user spends an average of two hours daily. So, it is easy for the companies to capture them in Facebook over any other forms of advertising.
Banner –ad revenue is expected to rise by 81%, which will result in decline of Google’s online advertising market share from 41% to 34 percent in the coming year. Forbes magazine has released the annual list of richest Americans, in which Mark Zuckerberg has pushed Larry Page behind. Mark Zuckerberg CEO and president of Facebook was in 35th position last year and has moved to 14th position with a net worth of $6.9 billion. Now that’s growth!! Larry Page who was in 11th position last year is in 15th position.
The war between the duos of Internet space has started and Larry Page took a counteroffensive defence to defend the market share by introducing Google plus. Mark zuckerberg’s recent announcement to go for IPO , also alarmed Google to retaliate in order to retain their position in Social web. Orkut which belongs to Google was once the most preferred social networking site, but Orkut rapidly faced a decline in the number of users after youngsters found FB to be more interesting as it had new features. Google also tried introducing google buzz which appeared to be a twitter clone trying to leverage the power of google. But buzz too was a failure.
""".split("\n")[1:-1]
# print(len(corpus))
# clearing and tokenizing
listWord = []
for i in range(len(corpus)):
    listWord.append(corpus[i].lower().split())

# l_A = corpus[0].lower().split()
# l_B = corpus[1].lower().split()
# l_C = corpus[2].lower().split()

# Calculating bag of words
#@@ tao ra chuoi cac tu khong trung nhau co trong doan van
# word_set = set(l_A).union(set(l_B)).union(set(l_C))
# print(len(word_set))
word_set = set(listWord[0])
for i in range(1, len(listWord)):
    word_set = word_set.union(set(listWord[i]))
print(word_set)
print("-------------------------------------------------------")

# loc word_set loai bo cac dau cau, cac so
charArray = ["\.", "\'", "\,", "\""]

# @@ tao tu dien
# word_dict_A = dict.fromkeys(word_set, 0)
# word_dict_B = dict.fromkeys(word_set, 0)
# word_dict_C = dict.fromkeys(word_set, 0)

listWordDict = []
for i in range(len(listWord)):
    listWordDict.append(dict.fromkeys(word_set, 0))
# print(len(listWordDict))

# @@ tinh so lan xuat hien cua cac tu co trong word_set trong tung doan
# for word in l_A:
#     word_dict_A[word] += 1
# for word in l_B:
#     word_dict_B[word] += 1
# for word in l_C:
#     word_dict_C[word] += 1

for i in range(len(listWord)):
    for word in listWord[i]:
        listWordDict[i][word] += 1
# for dict in listWordDict:
#     print(dict)

# @@ tao tu dien luu tan xuat xuat hien cua tu trong cau
def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count / sum_nk
    return tf

# tf_A = compute_tf(word_dict_A, l_A)
# tf_B = compute_tf(word_dict_B, l_B)
# tf_C = compute_tf(word_dict_C, l_C)

tf = []
for i in range(len(listWordDict)):
    tf.append(compute_tf(listWordDict[i], listWord[i]))
# for tfs in tf:
#     print(tfs)

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

# idf = compute_idf([word_dict_A, word_dict_B, word_dict_C])
idf = compute_idf(listWordDict)
# print(idf)

def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf

# tf_idf_A = compute_tf_idf(tf_A, idf)
# tf_idf_B = compute_tf_idf(tf_B, idf)
# tf_idf_C = compute_tf_idf(tf_C, idf)
tf_idf = []
for i in range(len(tf)):
    tf_idf.append(compute_tf_idf(tf[i], idf))
# for a in tf_idf:
#     print(a)

X = []
for tf_idf_s in tf_idf:
    temp = []
    for val in tf_idf_s.values():
        temp.append(val)
    X.append(temp)
print(X)
# for a in X:
#     print(a)
# kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# for cluster in kmeans.cluster_centers_:
#     print("-------------------------------------------------------------")
#     print(cluster)
#     print("-------------------------------------------------------------")
