import numpy as np
import matplotlib.pylab as plt


class FunctionMode(object):
    r = 0.05

    def __init__(self, hidden_size=100):
        self.params = {'W1': np.random.random((1, hidden_size)),
                       'B1': np.zeros(hidden_size),
                       'W2': np.random.random((hidden_size, 1)),
                       'B2': np.zeros(1)}

    @staticmethod
    def generate_data(fun, is_noise=True, axis=np.array([-1, 1, 100])):
        np.random.seed(0)
        x = np.linspace(axis[0], axis[1], axis[2])[:, np.newaxis]
        x_size = x.size
        y = np.zeros((x_size, 1))
        if is_noise:
            noise = np.random.normal(0, 0.1, x_size)
        else:
            noise = None

        for i in range(x_size):
            if is_noise:
                y[i] = fun(x[i]) + noise[i]
            else:
                y[i] = fun(x[i])

        return x, y

    @staticmethod
    def sigmoid(x_):
        return 1 / (1 + np.exp(-x_))

    def sigmoid_grad(self, x_):
        return (1.0 - self.sigmoid(x_)) * self.sigmoid(x_)

    def predict(self, x_):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['B1'], self.params['B2']

        a1 = np.dot(x_, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        return a2

    def loss(self, x_, t):
        y_ = self.predict(x_)
        return y_, np.mean((t - y_) ** 2)

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['B1'], self.params['B2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        # backward
        dy = (a2 - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['B2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = self.sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['B1'] = np.sum(da1, axis=0)
        return grads

    def train_with_own(self, x_, y_, max_steps=100):
        for k in range(max_steps):
            grad = self.gradient(x_, y_)
            for key in ('W1', 'B1', 'W2', 'B2'):
                self.params[key] -= self.r * grad[key]

            pred, loss = self.loss(x_, y_)
            if k % 150 == 0:
                plt.cla()
                plt.scatter(x_, y_)
                plt.plot(x_, pred, 'r-', lw=5)
                plt.text(0.5, 0, 'Loss=%.4f' % abs(loss), fontdict={'size': 20, 'color': 'red'})
                plt.pause(0.1)

        plt.ioff()
        plt.show()

    def validate(self, v_x):
        print(self.predict(v_x))


def func(x):
    return np.sin(x)


sin_mode = FunctionMode(150)
x, y = sin_mode.generate_data(func, False, np.array([-3, 3, 100]))
sin_mode.train_with_own(x, y, 5000)
