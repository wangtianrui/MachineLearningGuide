import numpy as np
import os


def loadData():
    origin_path = "./data"
    raw_strs = []
    lists = []
    pathes = ["sentence.txt", "sentencelabel.txt", "鲍骞月语录.txt"]
    for path in pathes:
        temp = os.path.join(origin_path, path)
        file = open(temp)
        raw_strs.append(file.read())
        file.close()

    list_temp = np.array(raw_strs[0].split("\n"))
    lists.append(list_temp)
    list_temp = np.array(raw_strs[1].split(";"), dtype=int)
    lists.append(list_temp)
    list_temp = np.array(raw_strs[2].split("\n"))
    lists.append(list_temp)
    return lists


def get_all_words_vec(train_list):
    word_set = set()
    for string in train_list:
        for word in string:
            word_set.add(word)
    return np.array(list(word_set))


def string_2_vec(string_list, words_vec):
    words_vec = np.array(words_vec)
    vec_list = []
    for string in string_list:
        vec = np.zeros(len(words_vec))
        for word in string:
            one_hot = (word == words_vec).astype(int)
            vec += one_hot
        vec_list.append(vec)
    # print(vec_list)
    return vec_list


def train(train_vec, label):
    """
    学习是1   不学习是0
    :param train_vec:
    :param label:
    :return:
    """
    train_vec = np.array(train_vec)
    label = np.array(label)
    p0_vec = np.zeros(len(train_vec[0]))
    p1_vec = np.zeros(len(train_vec[0]))

    p0_denominator = np.zeros(len(p0_vec))
    p1_denominator = np.zeros(len(p1_vec))
    for index in range(len(train_vec)):
        if label[index] == 0:
            p0_vec += train_vec[index]
            p0_denominator += 1
        if label[index] == 1:
            p1_vec += train_vec[index]
            p1_denominator += 1
    p0_vec = p0_vec / p0_denominator
    p1_vec = p1_vec / p1_denominator
    pStudy = np.sum(label) / len(label)
    return p0_vec, p1_vec, pStudy


def predict(x_vec, p0_vec, p1_vec, pstudy):
    p0 = np.mean(x_vec * p0_vec / (1 - pstudy))
    p1 = np.mean(x_vec * p1_vec / (pstudy))
    if p0 > p1:
        print("玩")
    else:
        print("学习")


if __name__ == "__main__":
    # print("hello")
    data_list = loadData()
    # print(data_list)
    train_list = data_list[0]
    labels = data_list[1]
    test_list = data_list[2]

    wordsVec = get_all_words_vec(train_list)
    # print(train_list,"\n",wordsVec)
    train_vec = string_2_vec(train_list, wordsVec)

    p0_vec, p1_vec, pstudy = train(train_vec, labels)
    # print(p0_vec, p1_vec, pstudy)
    test_vec = string_2_vec(test_list, wordsVec)
    for item in test_vec:
        predict(item, p0_vec, p1_vec, pstudy)
