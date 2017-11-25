#coding: utf-8

import sys
import os
import collections
import itertools
import operator
import array
import argparse
import numpy as np
import jieba
import sklearn
import sklearn.linear_model as linear_model


def fetch_train_test(data_path, test_size=0.2):
    """读取数据，分词并拆分数据为训练集和测试集
    """
    y = list()
    text_list = list()
    for line in open(data_path, "r").xreadlines():
        # 拆分为(X,Y)形式,X代表特征数据，Y代表标记
        label, text = line[:-1].split('\t', 1)
        # 使用jieba分词工具进行分词
        text_list.append(list(jieba.cut(text)))
        y.append(int(label))
    # 利用sklearn包划分数据集
    return sklearn.model_selection.train_test_split(
                text_list, y, test_size=test_size, random_state=1028)


def build_dict(text_list, min_freq=5):
    """根据传入的文本列表，创建一个最小频次为min_freq的字典，并返回字典word -> wordid
    """
    freq_dict = collections.Counter(itertools.chain(*text_list))
    freq_list = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    # 过滤低频词
    words, _ = zip(*filter(lambda wc: wc[1] >= min_freq, freq_list))
    return dict(zip(words, range(len(words))))


def text2vect(text_list, word2id):
    """将传入的文本转化为向量，返回向量大小为[n_samples, dict_size]
    """
    X = list()
    for text in text_list:
        vect = array.array('l', [0] * len(word2id))
        for word in text:
            if word not in word2id:
                continue
            vect[word2id[word]] = 1
        X.append(vect)
    return X


def evaluate(model, X, y):
    """评估数据集，并返回评估结果，包括：正确率、AUC值
    """
    accuracy = model.score(X, y)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, model.predict_proba(X)[:, 1], pos_label=1)
    return accuracy, sklearn.metrics.auc(fpr, tpr)
# 程序执行的入口！！
if __name__ == "__main__":
    
    # 原始数据集存放路径，这是一个有监督模型，数据中每一行的第一位为标记Y，Y=1代表该条是垃圾短信，Y=0则是正常短信
    data="train.txt"
    # step 1. 将原始数据构建训练集(X,Y)，分词并拆分成训练集train和测试集test
    X_train, X_test, y_train, y_test = fetch_train_test(data)

    # step 2. 创建字典
    word2id = build_dict(X_train, min_freq=10)

    # step 3. 抽取特征，文本特征的数值化，这里使用了Bag-of-words模型，常用的方式是还有TF-IDF、word2vec等
    X_train = text2vect(X_train, word2id)
    X_test = text2vect(X_test, word2id)

    # step 4. 训练模型，我们使用逻辑回归模型来解决这个二分类的问题，只用调用sklearn中封装好LR的模型,
    # 参数C控制了在正则化项L1/L2在最终的损失函数中所占的比重，详细说明自己查skilearn的官网API
    lr = linear_model.LogisticRegression(C=1)
    lr.fit(X_train, y_train)

    # step 5. 模型评估，至于模型的评估参数自己多查资料理解吧
    accuracy, auc = evaluate(lr, X_train, y_train)
    sys.stdout.write("训练集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("训练集AUC值：%.6f\n" % (auc))
    # 测试集上的评测结果
    accuracy, auc = evaluate(lr, X_test, y_test)
    sys.stdout.write("测试集正确率：%.4f%%\n" % (accuracy * 100))
    sys.stdout.write("测试AUC值：%.6f\n" % (auc))


'''
控制台输出：
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/nm/vrcffrqs4c374kp3jjx77d680000gn/T/jieba.cache
Loading model cost 0.681 seconds.
Prefix dict has been built succesfully.
训练集正确率：99.1114%
训练集AUC值：0.999267
测试集正确率：96.3098%
测试AUC值：0.990265
[Finished in 26.9s]
'''