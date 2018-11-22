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
        self.weight = 0.0
        self.biass = 0.0

        self.losses = []

    def predict(self, x):
        return self.weight * (x - 7) * (x + 1) * (x - 25)

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

            temp = (X_train - 7) * (X_train + 1) * (X_train - 25)

            y_pre = self.weight * temp

            loss_ = np.mean((y_label - y_pre) ** 2)
            self.losses.append(loss_)
            if i % 100 == 0:
                print("step:%d ; loss:%0.5f" % (i, loss_))

            dcom = -2 * (y_label - y_pre)
            dw = np.mean(temp * dcom) * learning_rate
            self.weight -= dw


class Stero:

    def __init__(self):
        self.alpha = 0.0
        self.beta = 0.0
        self.theta = 0.0
        self.bias = 0.0

        self.losses = []

    def predict(self, x):
        x = (x - np.mean(x)) / np.std(x)
        return self.alpha * (x ** 3) + self.beta * (x ** 2) + self.theta * x + self.bias

    def loss(self, X, y_true):
        y_pre = self.predict(X)
        loss = np.sum((y_true - y_pre) ** 2)
        return loss

    def drawer(self, X, y):
        plt.subplot(2, 1, 1)
        plt.scatter(X, y)
        X = np.arange(-1, 24, 0.1)
        y = self.predict(X)
        plt.plot(X, y)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()

    def train(self, X, y_true, steps, batch_size, learning_rate):
        X = np.array(X)
        y_true = np.array(y_true)

        for i in range(steps):
            index = np.random.randint(low=0, high=len(y_true), size=(batch_size))
            # index = np.random.randint(low=0, high=len(y_true), size=(5))
            X_train = X[index]
            y_label = y_true[index]
            X_train = (X_train - np.mean(X_train)) / np.std(X_train)
            y_pre = self.predict(X_train)
            if i % 100 == 0:
                y_ = self.predict(X_train)
                loss_ = np.mean((y_label - y_) ** 2)
                self.losses.append(loss_)
                print("step:%d ; loss:%0.5f" % (i, loss_))
            x_3 = (X_train ** 3)
            x_2 = (X_train ** 2)
            x_1 = X_train
            dcom = -2 * (y_label - y_pre)

            dalpha = np.mean(x_3 * dcom) * learning_rate
            dbeta = np.mean(x_2 * dcom) * learning_rate
            dtheta = np.mean(x_1 * dcom) * learning_rate
            dbias = np.mean(dcom) * learning_rate

            self.alpha -= dalpha
            self.beta -= dbeta
            self.theta -= dtheta
            self.bias -= dbias


class lwlr:

    def __init__(self, k):
        self.k = k
        self.weights = []
        self.predict_points = []

    def predict(self, testpoint, X, y):

        X = np.array(X)
        y = np.array(y)

        length = len(X)

        if len(X[0]) == 1:
            for i in range(length):
                X[i] = [i + 1, X[i]]

        print(X.shape)
        weight = np.eye(length)
        for j in range(length):
            diff = X[j] - testpoint
            weight[j, j] = np.exp(np.dot(diff, diff.T) / (-2.0 * self.k ** 2))

        xWx = np.dot(X.T, weight).dot(X)
        xWy = np.dot(X.T)
        thetas = np.linalg.inv(xWx).dot()
