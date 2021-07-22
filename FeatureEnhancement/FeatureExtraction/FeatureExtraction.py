'''
# DictVectorizer对使用字典存储的数据进行特征抽取与向量化
from sklearn.feature_extraction import DictVectorizer
# 定义一组字典列表，用来表示多个数据样本(每个字典代表一个数据样本)
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.}
]
# 初始化DictVectorizer特征提取器
vec = DictVectorizer()
# 输出转化后的特征矩阵
print(vec.fit_transform(measurements).toarray())
# 输出各个维度的特征含义
print(vec.get_feature_names())
'''


from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')
x_train, x_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.25, random_state=33)
# 使用CountVectorizer在去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试
count_vec = CountVectorizer(analyzer='word', stop_words='english')
# 只使用词频统计的方式将训练和测试文本转化为特征向量
x_count_train = count_vec.fit_transform(x_train)
x_count_test = count_vec.transform(x_test)
mnb_count = MultinomialNB()
mnb_count.fit(x_count_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes'
      '(CountVectorizer by filtering stopwords):',
      mnb_count.score(x_count_test, y_test))
y_count_predict = mnb_count.predict(x_count_test)
print('precision&recall&f1-score with CountVectorizer:', '\n',
      classification_report(y_test, y_count_predict,
                            target_names=news.target_names))


# 使用TfidfVectorizer在去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(analyzer='word', stop_words='english')
# 使用tfidf的方式将训练和测试文本转化为特征向量
x_tfidf_train = tfidf_vec.fit_transform(x_train)
x_tfidf_test = tfidf_vec.transform(x_test)
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(x_tfidf_train, y_train)
print('The accuracy of classifying 20newsgroups using Naive Bayes'
      '(TfidfTransformer by filtering stopwords):',
      mnb_tfidf.score(x_tfidf_test, y_test))
y_tfidf_predict = mnb_tfidf.predict(x_tfidf_test)
print('precision&recall&f1-score with TfidfTransformer:', '\n',
      classification_report(y_test, y_tfidf_predict,
                            target_names=news.target_names))

