from matplotlib import pyplot as plt
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


#each point is length, width, type(0, 1)

data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, .5, 1],
        [2, .5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]


mystery_flower = [4.5, 1]


w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()


# T = np.linspace(-5, 5, 10)
# Y = sigmoid(T)

#plt.plot(T, Y, c="r")
#plt.plot(T, sigmoid_prime(T), c="b" )
#plt.show()

#scatter data
# plt.axis([0,6,0,6])
# plt.grid()
#
# for i in range(len(data)):
#     point = data[i]
#     color = 'r'
#     if point[2] == 0:
#         color = 'b'
#     plt.scatter( point[0], point[1], c=color)
#
# plt.show()

learning_rate = 0.2
costs = []
#training loop
for i in range(50000):
    ri =  np.random.randint(len(data))
    point = data[ri]


    z = point[0] * w1 + point[1] * w2 + b
    prediction = sigmoid(z)

    target = point[2]
    cost = np.square(prediction - target)

    costs.append(cost)

    dcost_pred = 2 * (prediction - target)
    dpred_dz = sigmoid_prime(z)
    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db = 1

    dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
    dcost_dw2 = dcost_pred * dpred_dz * dz_dw2
    dcost_db = dcost_pred * dpred_dz * dz_db

    w1 = w1 - learning_rate * dcost_dw1
    w2 = w2 - learning_rate * dcost_dw2
    b = b - learning_rate * dcost_db


z = mystery_flower[0] * w1 + mystery_flower[1] * w2 + b
prediction = sigmoid(z)

print(prediction)
