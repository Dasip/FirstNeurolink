import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def calculate_mse(true, pred):
    return ((true - pred) ** 2).mean()


class Neuron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class NeuroNet:
    def __init__(self):
        # Веса
        # вход 1
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        # вход 2
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # веса вывода
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()

        # Пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        o1 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        - data - массив numpy (n x 2) numpy, n = к-во наблюдений в наборе.
        - all_y_trues - массив numpy с n элементами.
          Элементы all_y_trues соответствуют наблюдениям в data.
        '''
        learn_rate = 0.1
        epochs = 10000  # сколько раз пройти по всему набору данных

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Прямой проход (эти значения нам понадобятся позже)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w7 * h1 + self.w8 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Считаем частные производные.
                # --- Имена: d_L_d_w1 = "частная производная L по w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w7 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w8 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w7 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w8 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w6 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Обновляем веса и пороги
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_w7
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_w8
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Считаем полные потери в конце каждой эпохи
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = calculate_mse(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


data = np.array([
    [1, -1, 0],  # да нет
    [0, 1, 0],  # наверно да
    [1, -1, 0],  # конечно нет
    [0.5, -1, 0],  # точно нет
    [1, 0.5, 0],  # да определенно
    [0, 1, 0],  # ну да
    [0, -1, 0],  # ну нет
    [1, 0.5, 0],  # да определенно
    [1, -1, 0.5],  # да нет наверно
    [1, 0.5, -1],  # да точно нет
    [0.5, 0.5, -1],  # наверно точно нет
    [0, 0, 0], #
    [-1, 0, 0],  # отрицательный 1
    [-1, 0.5, 0], # полуопределенный отрицательный
    [0, -1, 0], # оттрицательны 2
    [0, 0, -1],  # оттрицательны 3
    [-1, 1, 0], # не да ...
])

all_y_trues = np.array([
    0,  # да нет
    1,  # наверно да
    0,  # конечно нет
    0,  # точно нет
    1,  # да определенно
    1,  # ну да
    0,  # ну нет
    1,  # да определенно
    0,  # да нет наверно
    0,  # да точно нет
    0,  # наверно точно нет
    0.5, # неопределенный ответ
    0, # отрицательный 1
    0, # полуопределенный отрицательный
    0,  # оттрицательны2
    0, # оттрицательны 3
    0, # не да ...
])




if __name__ == "__main__":
    network = NeuroNet()
    network.train(data, all_y_trues)


    emily = np.array([1, 1, 0]) # ага да
    frank = np.array([-1, -1, 0])  # никак нет
    print("Ага да: %.3f" % network.feedforward(emily)) # 0.951 - Ж
    print("Никак нет: %.3f" % network.feedforward(frank)) # 0.039 - М

    emily = np.array([1, -1, 0.5]) # да нет наверно
    frank = np.array([0, 0, 1])  # в принципе да
    print("да нет наверно: %.3f" % network.feedforward(emily)) # 0.951 - Ж
    print("в принципе да: %.3f" % network.feedforward(frank)) # 0.039 - М