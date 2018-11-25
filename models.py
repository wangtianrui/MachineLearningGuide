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

    def predict(self, x, isPre=False):
        if isPre:
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
        y = self.predict(X, isPre=True)
        plt.plot(X, y)

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()

    def train(self, X, y_true, steps, batch_size, learning_rate):
        X = np.array(X)
        y_true = np.array(y_true)
        X = (X - np.mean(X)) / np.std(X)
        for i in range(steps):
            index = np.random.randint(low=0, high=len(y_true), size=(batch_size))
            # index = np.random.randint(low=0, high=len(y_true), size=(5))
            X_train = X[index]
            y_label = y_true[index]
            # X_train = (X_train - np.mean(X_train)) / np.std(X_train)
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

    def __init__(self):
        self.weights = []
        self.predict_points = []

    def predict(self, testpoint, X, y, k):
        X = np.array(X)
        y = np.array(y)

        size = len(X)
        weight = np.eye(size)

        for i in range(size):
            diff = X[i] - testpoint
            weight[i, i] = np.exp(diff ** 2 / (-2 * k ** 2))
        xWx = X.dot(weight).dot(X.T)

        if xWx == 0.0:
            print("This matrix is singular, cannot do inverse")
            print(testpoint)
            return
        theta = ((1.0 / xWx) * X).dot(weight).dot(y.T)

        print(theta)
        return theta

    def drawer(self, X, y, predicts):
        # X = [x[1] for x in X]
        y_pre = np.array(predicts)
        plt.scatter(X, y)
        print(y_pre)
        plt.plot(X, y_pre.flatten())
        plt.show()


class Bayes:
    def __init__(self):
        self.p0vec = []
        self.p1vec = []
