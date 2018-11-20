import numpy as np
import matplotlib.pyplot as plt


class OrdinaryLinear:
    """
     y = wx + b
    """

    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0
        self.losses = []

    def predict(self, x):
        return self.weight * x + self.bias

    def loss(self, X, y_true):
        y_pre = self.predict(X)
        loss = np.sum((y_true - y_pre) ** 2)
        return loss

    def drawer(self, X, y):
        plt.scatter(X, y)
        X = np.arange(-1, 24, 0.1)
        y = self.weight * X + self.bias
        plt.plot(X, y)
        plt.show()

    def train(self, X, y_true, steps, batch_size, learning_rate):
        X = np.array(X)
        y_true = np.array(y_true)

        for i in range(steps):
            index = np.random.randint(low=0, high=len(y_true), size=(batch_size))
            # index = np.random.randint(low=0, high=len(y_true), size=(5))
            X_train = X[index]
            y_label = y_true[index]
            y_pre = self.predict(X_train)
            loss_ = np.mean((y_label - y_pre) ** 2)
            self.losses.append(loss_)
            print("step:%d ; loss:%0.5f" % (i, loss_))
            dw = np.mean(-2 * X_train * (y_label - y_pre)) * learning_rate
            db = np.mean(-2 * (y_label - y_pre)) * learning_rate

            self.weight -= dw
            self.bias -= db


class Multinomial:
    """
    y = a * x^2 + b * x + bias
    """

    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.bias = 0.0

        self.gama = 0.0
        self.sin_a = 1.0
        self.sin_b = 1.0

        self.log_weight = 1.0
        self.log_a = 1.0
        self.log_b = 1.0
        self.losses = []

    def predict(self, x):
        return self.alpha * (x ** 2) + self.beta * x + self.bias \
               + self.gama * np.sin(self.sin_a * x + self.sin_b) + self.log_weight * np.log(
            self.log_a * (x + 1) + self.log_b)

    def drawer(self, X, y):
        plt.scatter(X, y)
        X = np.arange(0, 24, 0.1)
        y = self.predict(X)
        plt.plot(X, y)
        plt.show()

    def train(self, X, y_true, steps, batch_size, learning_rate):
        X = np.array(X)
        y_true = np.array(y_true)

        for i in range(steps):
            index = np.random.randint(low=0, high=len(y_true), size=(batch_size))
            # index = np.random.randint(low=0, high=len(y_true), size=(5))
            X_train = X[index]
            y_label = y_true[index]
            y_pre = self.predict(X_train)
            loss_ = np.mean((y_label - y_pre) ** 2)
            self.losses.append(loss_)
            print("step:%d ; loss:%0.5f" % (i, loss_))

            d_alpha = np.mean(-2 * (X_train ** 2) * (y_label - y_pre)) * learning_rate
            d_beta = np.mean(-2 * X_train * (y_label - y_pre)) * learning_rate
            d_bias = np.mean(-2 * (y_label - y_pre)) * learning_rate

            d_gama = np.mean(-2 * np.sin(self.sin_a * X_train + self.sin_b) * (y_label - y_pre)) * learning_rate
            d_sin_a = np.mean(
                -2 * self.gama * np.cos(self.sin_a * X_train + self.sin_b) * X_train * (
                        y_label - y_pre)) * learning_rate
            d_sin_b = np.mean(
                -2 * self.gama * np.cos(self.sin_a * X_train + self.sin_b) * (
                        y_label - y_pre)) * learning_rate

            d_log_weight = np.mean(-2 * (y_label - y_pre) * np.log(self.log_a * X_train + self.log_b)) * learning_rate
            d_log_a = np.mean(-2 * (y_label - y_pre) * self.log_weight * (
                    X_train / (self.log_a * X_train + self.log_b))) * learning_rate
            d_log_b = np.mean(-2 * (y_label - y_pre) * self.log_weight * (
                    1 / (self.log_a * X_train + self.log_b))) * learning_rate
            # print(d_gama,d_sin_a,d_sin_b)
            self.alpha -= d_alpha
            self.beta -= d_beta
            self.bias -= d_bias
            self.gama -= d_gama
            self.sin_a -= d_sin_a
            self.sin_b -= d_sin_b

            self.log_weight -= d_log_weight
            self.log_a -= d_log_a
            self.log_b -= d_log_b
            # print(d_log_a,d_log_b,d_log_weight)

            # print("d_gama:%d" % (d_gama))


class Sin:
    """
     y = wx + b
    """

    def __init__(self):
        self.weight = 1.0
        self.bias = 1.0
        self.x_a = 1.0
        self.x_b = 1.0
        self.losses = []

    def predict(self, x):
        return self.weight * np.sin(self.x_a * x + self.x_b) + self.bias

    def loss(self, X, y_true):
        y_pre = self.predict(X)
        loss = np.sum((y_true - y_pre) ** 2)
        return loss

    def drawer(self, X, y):
        plt.scatter(X, y)
        X = np.arange(-1, 24, 0.1)
        y = self.predict(X)
        plt.plot(X, y)
        plt.show()

    def train(self, X, y_true, steps, batch_size, learning_rate):
        X = np.array(X)
        y_true = np.array(y_true)

        for i in range(steps):
            index = np.random.randint(low=0, high=len(y_true), size=(batch_size))
            # index = np.random.randint(low=0, high=len(y_true), size=(5))
            X_train = X[index]
            y_label = y_true[index]
            y_pre = self.predict(X_train)
            loss_ = np.mean((y_label - y_pre) ** 2)
            self.losses.append(loss_)
            if i % 100 == 0:
                print("step:%d ; loss:%0.5f" % (i, loss_))

            dcom = -2 * (y_label - y_pre)

            dw = np.mean(dcom * np.sin(self.x_a * X_train + self.x_b)) * learning_rate
            db = np.mean(dcom) * learning_rate

            dx_a = np.mean(self.weight * dcom * np.cos(self.x_a * X_train + self.x_b) * X_train) * learning_rate
            dx_b = np.mean(self.weight * dcom * np.cos(self.x_a * X_train + self.x_b)) * learning_rate

            self.weight -= dw
            self.bias -= db
            self.x_a -= dx_a
            self.x_b -= dx_b


class Net:

    def __init__(self):
        self.weight_1 = 1.0
        self.bias_1 = 1.0
        self.weight_2 = 1.0
        self.bias_2 = 1.0

        self.losses = []

    def sigmod(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, x):
        first = self.weight_1 * x + self.bias_1
        return self.weight_2 * self.sigmod(first) + self.bias_2

    def loss(self, X, y_true):
        y_pre = self.predict(X)
        loss = np.sum((y_true - y_pre) ** 2)
        return loss

    def drawer(self, X, y):
        plt.scatter(X, y)
        X = np.arange(-1, 24, 0.1)
        y = self.predict(X)
        plt.plot(X, y)
        plt.show()

    def train(self, X, y_true, steps, batch_size, learning_rate):
        X = np.array(X)
        y_true = np.array(y_true)

        for i in range(steps):
            index = np.random.randint(low=0, high=len(y_true), size=(batch_size))
            # index = np.random.randint(low=0, high=len(y_true), size=(5))
            X_train = X[index]
            y_label = y_true[index]

            first = self.weight_1 * X_train + self.bias_1
            logist = self.sigmod(first)
            y_pre = self.weight_2 * logist + self.bias_2

            loss_ = np.mean((y_label - y_pre) ** 2)
            self.losses.append(loss_)
            if i % 100 == 0:
                print("step:%d ; loss:%0.5f" % (i, loss_))

            dcom = -2 * (y_label - y_pre)

            dw2 = np.mean(dcom * logist) * learning_rate
            db2 = np.mean(dcom) * learning_rate

            dw1 = np.mean(dcom * logist * (1 - logist) * self.weight_2 * X_train) * learning_rate
            db1 = np.mean(dcom * logist * (1 - logist) * self.weight_2) * learning_rate

            self.weight_1 -= dw1
            self.weight_2 -= dw2
            self.bias_1 -= db1
            self.bias_2 -= db2


class Three:

    def __init__(self):
        self.weight_1 = 1.0
        self.bias_1 = 1.0
        self.weight_2 = 1.0
        self.bias_2 = 1.0
        self.weight_3 = 1.0
        self.bias_3 = 1.0

        self.losses = []

    def sigmod(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, x):
        first = self.weight_1 * x + self.bias_1
        second = self.weight_2 * self.sigmod(first) + self.bias_2
        return self.weight_3 * self.sigmod(second) + self.bias_3

    def loss(self, X, y_true):
        y_pre = self.predict(X)
        loss = np.sum((y_true - y_pre) ** 2)
        return loss

    def drawer(self, X, y):
        plt.scatter(X, y)
        X = np.arange(-1, 24, 0.1)
        y = self.predict(X)
        plt.plot(X, y)
        plt.show()

    def train(self, X, y_true, steps, batch_size, learning_rate):
        X = np.array(X)
        y_true = np.array(y_true)

        for i in range(steps):
            index = np.random.randint(low=0, high=len(y_true), size=(batch_size))
            # index = np.random.randint(low=0, high=len(y_true), size=(5))
            X_train = X[index]
            y_label = y_true[index]

            first = self.weight_1 * X_train + self.bias_1
            logist_1 = self.sigmod(first)
            second = self.weight_2 * logist_1 + self.bias_2
            logist_2 = self.sigmod(second)
            y_pre = self.weight_3 * logist_2 + self.bias_3

            loss_ = np.mean((y_label - y_pre) ** 2)
            self.losses.append(loss_)
            if i % 100 == 0:
                print("step:%d ; loss:%0.5f" % (i, loss_))

            dcom = -2 * (y_label - y_pre)

            dw3 = np.mean(dcom * logist_2) * learning_rate
            db3 = np.mean(dcom) * learning_rate

            dw2 = np.mean(dcom * logist_2 * self.weight_3 * (1 - logist_2) * first) * learning_rate
            db2 = np.mean(dcom * logist_2 * self.weight_3 * (1 - logist_2)) * learning_rate

            dw1 = np.mean(
                dcom * logist_2 * (1 - logist_2) * self.weight_3 * self.weight_2 * logist_1 * (
                        1 - logist_1) * X_train) * learning_rate
            db1 = np.mean(dcom * logist_2 * self.weight_3 * (1 - logist_2) * logist_1 * (1 - logist_1)) * learning_rate

            self.weight_1 -= dw1
            self.bias_1 -= db1
            self.weight_2 -= dw2
            self.bias_2 -= db2
            self.weight_3 -= dw3
            self.bias_3 -= db3


class lwlr:

    def __init__(self, k):
        self.k = k
        self.weights = []

    def predict(self, testpoint, X, y):
        X = np.array(X)
        y = np.array(y)
        length = len(X)
        weight = []

        for j in range(length):
            diff = X[j] - testpoint
            weight[j] = np.exp(np.dot(diff, diff.T) / (-2.0 * self.k ** 2))

        thetas =  weight 
