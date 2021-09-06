from ctypes import *
import  numpy as  np
import ctypes
from numpy.ctypeslib import ndpointer 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


lib = CDLL('./SGD.so')   

lib.connect()

def SGD(max_epoch, iterate_count, learning_rate, n, m, train_x, train_y, weights):
    lib.stochastic_gradient_descent.argtypes = [c_int, c_int, c_double, c_int, c_int, c_void_p, c_void_p, c_void_p]
    return lib.stochastic_gradient_descent(max_epoch, iterate_count, learning_rate, n, m, train_x.ctypes.data_as(c_void_p), train_y.ctypes.data_as(c_void_p), weights.ctypes.data_as(c_void_p))

def SVRG(max_epoch, iterate_count, learning_rate, n, m, train_x, train_y, weights):
    lib.stochastic_variance_reduced_gradient.argtypes = [c_int, c_int, c_double, c_int, c_int, c_void_p, c_void_p, c_void_p]
    return lib.stochastic_variance_reduced_gradient(max_epoch, iterate_count, learning_rate, n, m, train_x.ctypes.data_as(c_void_p), train_y.ctypes.data_as(c_void_p), weights.ctypes.data_as(c_void_p))

def SARAH(max_epoch, iterate_count, learning_rate, n, m, train_x, train_y, weights):
    lib.SARAH.argtypes = [c_int, c_int, c_double, c_int, c_int, c_void_p, c_void_p, c_void_p]
    return lib.SARAH(max_epoch, iterate_count, learning_rate, n, m, train_x.ctypes.data_as(c_void_p), train_y.ctypes.data_as(c_void_p), weights.ctypes.data_as(c_void_p))

def signSGD(max_epoch, iterate_count, learning_rate, beta, n, m, train_x, train_y, weights):
    lib.signSGD.argtypes = [c_int, c_int, c_double, c_double, c_int, c_int, c_void_p, c_void_p, c_void_p]
    return lib.signSGD(max_epoch, iterate_count, learning_rate, beta, n, m, train_x.ctypes.data_as(c_void_p), train_y.ctypes.data_as(c_void_p), weights.ctypes.data_as(c_void_p))

def ADAM(max_epoch, iterate_count, learning_rate, beta_1, beta_2, epsilon, n, m, train_x, train_y, weights):
    lib.ADAM.argtypes = [c_int, c_int, c_double, c_double, c_double, c_double, c_int, c_int, c_void_p, c_void_p, c_void_p]
    return lib.ADAM(max_epoch, iterate_count, learning_rate, beta_1, beta_2, epsilon, n, m, train_x.ctypes.data_as(c_void_p), train_y.ctypes.data_as(c_void_p), weights.ctypes.data_as(c_void_p))


train_data_x = np.genfromtxt("test_data/Iris_train_data_x.csv", delimiter=',')
train_data_y = np.genfromtxt("test_data/Iris_train_data_y.csv", delimiter=',')


test_data_x = np.genfromtxt("test_data/Iris_test_data_x.csv", delimiter=',')
test_data_y = np.genfromtxt("test_data/Iris_test_data_y.csv", delimiter=',')

m = train_data_x.shape[1]

lib.stochastic_gradient_descent.restype = ndpointer(dtype=c_double, shape=(m,))
lib.stochastic_variance_reduced_gradient.restype = ndpointer(dtype=c_double, shape=(m,))
lib.SARAH.restype = ndpointer(dtype=c_double, shape=(m,))
lib.signSGD.restype = ndpointer(dtype=c_double, shape=(m,))
lib.ADAM.restype = ndpointer(dtype=c_double, shape=(m,))

def accuracy(a, data_x, data_y):
    return 100 - (100 * np.sum(abs(np.round(sigmoid(data_x @ a)) - data_y)/data_x.shape[0]))


weights_s = np.array([1, 1, 1, 1])
weights_sign = np.array([1, 1, 1, 1])
weights_adam = np.array([1, 1, 1, 1])
weights_sarah = np.array([1, 1, 1, 1])
weights_svrg = np.array([1, 1, 1, 1])

signSGD_ = []
ADAM_ = []
SARAH_ = []
SGD_ = []
SVRG_ = []
i_ = []
j_ = []

# i = 20
j = 40
#for j in range(10):
for i in range(500):
	i_.append(i)
	j_.append(j)
	SGD_.append(accuracy(SGD(i, j, 0.001, train_data_x.shape[0], train_data_x.shape[1], train_data_x, train_data_y, weights_s), test_data_x, test_data_y))
	SARAH_.append(accuracy(SARAH(i, j, 0.001, train_data_x.shape[0], train_data_x.shape[1], train_data_x, train_data_y, weights_sarah), test_data_x, test_data_y))
	SVRG_.append(accuracy(SVRG(i, j, 0.001, train_data_x.shape[0], train_data_x.shape[1], train_data_x, train_data_y, weights_svrg), test_data_x, test_data_y))
	ADAM_.append(accuracy(ADAM(i, j, 0.001, 0.9, 0.999, 1e-8, train_data_x.shape[0], train_data_x.shape[1], train_data_x, train_data_y, weights_adam), test_data_x, test_data_y))
	signSGD_.append(accuracy(signSGD(i, j, 0.001, 0.6, train_data_x.shape[0], train_data_x.shape[1], train_data_x, train_data_y, weights_sign), test_data_x, test_data_y))



plt.plot(i_, signSGD_,label='signSGD')
plt.plot(i_, SGD_,label='SGD')
plt.plot(i_, SARAH_,label='SARAH')
plt.plot(i_, SVRG_,label='SVRG')
plt.plot(i_, ADAM_,label='ADAM')
plt.legend()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(i_, j_, signSGD_)
# plt.xlabel("Epochs")
# plt.ylabel("Iterate count")

plt.show()

# print(accuracy(signSGD(20, 3, 0.01, 0.5, train_data_x.shape[0], train_data_x.shape[1], train_data_x, train_data_y, weights)))