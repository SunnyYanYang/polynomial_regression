import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# lin_reg_2 = LinearRegression()
# lin_reg_2.fit(X_poly, y)
def learn_curve(train_error_iteration, valid_error_iteration):
    train = np.array(train_error_iteration)
    valid = np.array(valid_error_iteration)
    train_error = train[:, 0]
    train_iteration = train[:, 1]
    valid_error = valid[:, 0]
    valid_iteration = valid[:, 1]
    plt.plot(train_iteration, train_error, '--r', valid_iteration, valid_error, 'b')
    plt.legend(['Train', 'Valid'])

    ax1 = plt.gca()
    ax1.set_title('learning curve')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('error')
    plt.show()


def approximate_gradient(x, y, a, lamda):
    approx_gradient = np.zeros((np.shape(x)[1], 1))
    for i in range(np.shape(x)[1]):
        b = np.zeros((np.shape(x)[1], 1), dtype=float)
        delta = 0.001
        b[i] = delta
        approx_gradient[i] = (compute_error_for_given_points(a + b, x, y, lamda)
                              - compute_error_for_given_points(a - b, x, y, lamda)) / (2 * delta)
    return -approx_gradient  # 近似梯度


def step_gradient(a_current, x, y, learning_rate, lamda):
    N = float(len(y))
    a_gradient = (1 / N) * np.matmul(x.T, np.matmul(x, a_current) - y) + lamda / N * a_current  # 梯度
    a_gradient[0] = a_gradient[0] - lamda / N * a_current[0]
    # print('a_gradient=\n', a_gradient.T, '\n', 'approximate_gradient=\n', approximate_gradient(x, y, a_current).T)
    new_a = a_current - (a_gradient * learning_rate)
    return new_a


def compute_error_for_given_points(a, x, y, lamda):
    total_error = (sum(np.power(y - np.matmul(x, a), 2)) + lamda * sum(np.power(a, 2))) / (2 * float(len(y)))  # 方差+正则化项
    return total_error


def gradient_descent_runner(x, y, starting_a, learning_rate, num_iterations, lamda, train_error_iteration, valid_x,
                            valid_y, validation_error_iteration):
    print("Starting gradient descent at a = \n{0}\n error = {1}"
          .format(starting_a.T, compute_error_for_given_points(starting_a, x, y, lamda)))
    print("Running...\n")
    a = starting_a
    for i in range(num_iterations):  # 这个部分要改成当误差差小于一定值之后，并且持续一定代数
        a = step_gradient(np.array(a), np.array(x), np.array(y), learning_rate, lamda)
        if i % 1 == 0:  # 每经过n次迭代记录该迭代的误差
            train_error_iteration.append([compute_error_for_given_points(a, x, y, lamda)[0], i])
            validation_error_iteration.append([compute_error_for_given_points(a, valid_x, valid_y, lamda)[0], i])
        if i % 100000 == 0:
            print("error", i + 1, "=", compute_error_for_given_points(a, x, y, lamda))
    print("After {0} iterations error = {1}\na = {2}"
          .format(num_iterations, compute_error_for_given_points(a, x, y, lamda), a.T))
    return [a, train_error_iteration, validation_error_iteration]


def gradient_descent_runner_delta_error(x, y, starting_a, learning_rate, lamda, train_error_iteration,
                                        valid_x, valid_y, validation_error_iteration):
    current_train_error = 0.
    current_valid_error = 0.
    num_iterations = 0
    stable_iteration = 20
    stable = 0
    if_fun_is_unfinished = 1
    threshold = 0.0001

    print("Starting gradient descent at a = \n{0}\n error = {1}"
          .format(starting_a.T, compute_error_for_given_points(starting_a, x, y, lamda)))
    print("Running...\n")
    a = starting_a
    while if_fun_is_unfinished:
        previous_a = a
        a = step_gradient(np.array(a), np.array(x), np.array(y), learning_rate, lamda)
        num_iterations += 1
        current_a = a
        delta_error = compute_error_for_given_points(current_a, x, y, lamda)-compute_error_for_given_points(previous_a, x, y, lamda)
        delta_error = abs(delta_error)
        current_train_error = compute_error_for_given_points(a, x, y, lamda)[0]
        current_valid_error = compute_error_for_given_points(a, valid_x, valid_y, lamda)[0]
        if num_iterations % 10 == 0:  # 每经过n次迭代记录该迭代的误差
            train_error_iteration.append([current_train_error, num_iterations])
            validation_error_iteration.append([current_valid_error, num_iterations])
        if num_iterations % 100000 == 0:
            print("error", num_iterations, "=", current_train_error)

        if delta_error <= threshold:  # 判断学习曲线是否稳定
            stable += 1
            if stable == stable_iteration:
                break
        elif delta_error > threshold and stable != 0:
            stable = 0
    print("After {0} iterations, train_error = {1} with max delta_error {2} in the next {3} iterations\n"
          "at the same time vilid_error ={4}\na = {5}"
          .format(num_iterations-stable_iteration, current_train_error,
                  threshold, stable_iteration, current_valid_error, a.T))
    return [a, train_error_iteration, validation_error_iteration]


def normalization(x, y):
    x_norm_constant = np.zeros((np.shape(x)[1], 1))
    for i in range(np.shape(x)[1]):
        x_norm_constant[i] = sum(x[:, i]) / float(len(y))
    x = x / x_norm_constant.T
    y_norm_constant = sum(y) / float(len(y))
    y = y / y_norm_constant
    # print('x={0}\ny={1}'.format(x, y))
    return [x, y, x_norm_constant, y_norm_constant]


def trans_normalization(a, x_norm_constant, y_norm_constant):
    a = y_norm_constant * a * x_norm_constant
    return a


def run():
    dataset = pd.read_csv('AFM_data_12.csv')
    feature_col = [6, 7, 8]  # 特征值所在列（从0开始数）
    target_col = [9]  # 目标值所在列（从0开始数）
    num_sample = 10000  # 样本数量
    X = dataset.iloc[0:num_sample, feature_col].values
    y = dataset.iloc[0:num_sample, target_col].values
    number_of_degree = 4
    X_poly = PolynomialFeatures(degree=number_of_degree).fit_transform(X)  # 极化X，其中degree为项最高次数
    [X_poly, y, x_norm_constant, y_norm_constant] = normalization(X_poly, y)

    valid_x = dataset.iloc[num_sample:num_sample * 2, feature_col].values
    valid_y = dataset.iloc[num_sample:num_sample * 2, target_col].values
    vaild_x_poly = PolynomialFeatures(degree=number_of_degree).fit_transform(valid_x)
    [valid_x_poly, valid_y, valid_x_norm_constant, valid_y_norm_constant] = normalization(vaild_x_poly, valid_y)

    train_error_iteration = []
    validation_error_iteration = []
    # print('X_poly={0}\ny={1}'.format(X_poly, y))
    learning_rate = 0.05  # 学习率
    lamda = 0.1
    initial_a = (np.random.rand(np.shape(X_poly)[1], 1) - 0.5 * np.ones((np.shape(X_poly)[1], 1))) * 200
    # np.array([[-19.64026234], [58.28010791], [-53.80700526], [16.1582763]])
    # [[-2.11340896  7.0904723  -5.08469312  1.10068242]]
    num_iterations = 100  # 迭代次数
    # 求优化后的归一化的a
    [norm_a, train_error_iteration, validation_error_iteration] = \
        gradient_descent_runner_delta_error(X_poly, y, initial_a, learning_rate, lamda, train_error_iteration,
                                            valid_x_poly, valid_y, validation_error_iteration)
    # [norm_a, train_error_iteration, validation_error_iteration] = gradient_descent_runner(X_poly, y, initial_a,
    #                                                                                       learning_rate, num_iterations,
    #                                                                                       lamda, train_error_iteration,
    #                                                                                       valid_x_poly, valid_y,
    #                                                                                       validation_error_iteration)
    # 画一个error-iteration(train / validation)的图像
    learn_curve(train_error_iteration, validation_error_iteration)

    # 画一个error-No. sample 图像
    a = trans_normalization(norm_a, x_norm_constant, y_norm_constant)


if __name__ == '__main__':
    run()
