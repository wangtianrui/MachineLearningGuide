import numpy as np
import os


def readdata():
    sentence_path = "./data/sentence.txt"
    label_path = "./data/sentencelabel.txt"
    bao_path = "./data/鲍骞月语录.txt"

    f = open(sentence_path)
    sentences = f.read()
    # print(sentences)
    f.close()

    f = open(label_path)
    sentences_label = f.read()
    # print(sentences_label)
    f.close()

    f = open(bao_path)
    sentences_erzi = f.read()
    # print(sentences_erzi)
    f.close()

    sentences = np.array(sentences.split("\n"))
    print(sentences)
    sentences_label = np.array(sentences_label.split(";"), dtype=int)
    print(sentences_label)
    sentences_erzi = np.array(sentences_erzi.split("\n"))
    print(sentences_erzi)
    return sentences, sentences_label, sentences_erzi


def stringlist2wordslist(stringList):
    wordSet = set()
    for string in stringList:
        for word in string:
            wordSet.add(word)
    return list(wordSet)


def string2vec(stringList, words_list):
    words_list = np.array(words_list)
    vec = np.zeros((len(stringList), len(words_list)))
    for index in range(len(stringList)):
        for word in stringList[index]:
            vec[index] += np.array((words_list == word), dtype=int)
            # print(vec[index])
    return vec


def train(train_vec, wordsVec, label):
    pAbuse = np.sum(label) / len(label)

    p0num = np.zeros(len(wordsVec))  # 骂人的
    p1num = np.zeros(len(wordsVec))

    p0denom = 0.0  # 字的总数，用于算 某个字在这个段话里出现的概率
    p1denom = 0.0

    for index in range(len(train_vec)):
        if label[index] == 0:
            p0num += train_vec[index]
            p0denom += np.sum(train_vec[index])
        elif label[index] == 1:
            p1num += train_vec[index]
            p1denom += np.sum(train_vec[index])

    p1_vec = p1num / p1denom
    p0_vec = p0num / p0denom

    return p0_vec, p1_vec, pAbuse


def predict_string2vec(stringlist, words_list):
    words_list = np.array(words_list)
    vec = np.zeros((len(stringlist), len(words_list)))
    for index in range(len(stringlist)):
        for word in stringlist[index]:
            vec[index] += np.array((words_list == word), dtype=int)
    return vec


def predict(vec, p0Vec, p1Vec, pA):
    p1 = np.sum(vec * p1Vec) * (1 - pA)
    p0 = np.sum(vec * p0Vec) * pA
    if p1 >= p0:
        print("文明")
    elif p1 < p0:
        print("脏话")




if __name__ == "__main__":
    train_list, label, test_list = readdata()
    words_list = stringlist2wordslist(train_list)
    # print(words_list)
    train_vec = string2vec(train_list, words_list)
    p0vec, p1vec, pabuse = train(train_vec, words_list, label)
    # print(p0vec, p1vec, pabuse)
    test_vec = predict_string2vec(test_list, words_list)
    print(test_vec)
    for vec in test_vec:
        predict(vec, p0vec, p1vec, pabuse)