import numpy as np
import os
import matplotlib.pyplot as plt
from models import OrdinaryLinear
from models import Multinomial
from models import Three
from models import Stero
from models import lwlr


def readData():
    path = "./data/temperature.txt"
    # path = "./data/linear_data.txt"
    if os.path.exists(path):
        file = open(path, "r")
        string = file.read()
        file.close()
    else:
        print("路径没有找到")
    return string


def dealData(string):
    string_list = string.split("，")
    number_list = np.array(string_list, dtype=int)
    return number_list


def drawer(data):
    x = range(len(data))
    plt.scatter(x, data)
    plt.show()


def ordinary_linear(X, y_true):
    model = OrdinaryLinear()
    model.train(X=X, y_true=y_true, steps=10000, batch_size=1, learning_rate=1e-5)
    model.drawer(X, y_true)


def multinomial(X, y_true):
    model = Multinomial()
    model.train(X, y_true, 140000, 10, 1e-5)
    model.drawer(X, y_true)



def three(X, y_true):
    model = Three()
    model.train(X, y_true, 500000, 5, 1e-7)
    model.drawer(X, y_true)


def stero(X, y_true):
    model = Stero()
    model.train(X, y_true, 300000, 20, 1e-5)
    model.drawer(X, y_true)


def lw(X, y_true):
    model = lwlr()
    predict = np.zeros(24)
    for index in range(len(y_true)):
        point = X[index]
        # print(point)
        theta = model.predict(point, X, y_true, 2)
        predict[index] = point * theta
    model.drawer(X, y_true, predict)


if __name__ == "__main__":
    origin_data = readData()
    number_data = dealData(origin_data)
    drawer(number_data)
    # X = np.array(range(len(number_data))).reshape((len(number_data), 1))
    X = np.array(range(len(number_data)))
    # ordinary_linear(X, number_data)
    # multinomial(X, number_data)
    # sin(X, number_data)
    # net(X, number_data)
    # three(X, number_data)
    stero(X, number_data)
    # one = np.ones((len(X), 1))
    # # X = np.concatenate((one, X), axis=1)
    # number_data.flatten()
    # X += 1
    # lw(X, number_data)
