import numpy as np
import os
import matplotlib.pyplot as plt
from models import OrdinaryLinear
from models import Multinomial
from models import Sin
from models import Net
from models import Three


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


def sin(X, y_true):
    model = Sin()
    model.train(X, y_true, 500000, 20, 1e-3)
    model.drawer(X, y_true)


def net(X, y_true):
    model = Net()
    model.train(X, y_true, 500000, 10, 1e-5)
    model.drawer(X, y_true)

def three(X, y_true):
    model = Three()
    model.train(X, y_true, 500000, 10, 1e-5)
    model.drawer(X, y_true)



if __name__ == "__main__":
    origin_data = readData()
    number_data = dealData(origin_data)
    drawer(number_data)
    X = range(len(number_data))
    # ordinary_linear(X, number_data)
    # multinomial(X, number_data)
    # sin(X, number_data)
    # net(X, number_data)
    three(X, number_data)
