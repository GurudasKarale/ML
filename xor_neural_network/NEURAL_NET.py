import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
label = np.array([[0],[1],[1],[0]])
learning_rate = 0.1

w1 = np.random.uniform(size=(2,2))
b1 =np.random.uniform(size=(1,2))
w2 = np.random.uniform(size=(2,1))
b2 = np.random.uniform(size=(1,1))

def multip(x,w1):
    g=np.zeros((x.shape[0],w1.shape[1]))
    for i in range(len(x)):
        for k in range(w1.shape[1]):
            summ=0

            for j in range(x.shape[1]):
                summ+=(x[i][j]*w1[j][k])
            g[i][k]=summ
    return g

def sigmoid (x):
    return 1/(1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

for i in range(10000):

    z1 = multip(x,w1) + b1
    a1 = sigmoid(z1)
    z2 = multip(a1,w2) + b2
    a2 = sigmoid(z2)

    error = (label - a2)
    deltaK = error * derivative(a2)  #derivative of output error = (expected - output) * transfer_derivative(output)

    deltaJ = multip(deltaK,w2.T)
    deltaJ = deltaJ * derivative(a1) #error = (weight_k * error_j) * transfer_derivative(hidden layer i.e a1)

    w2 += multip(a1.T,deltaK) * learning_rate
    b2 += np.sum(deltaK) * learning_rate
    w1 += multip(x.T,deltaJ) * learning_rate
    b1 += np.sum(deltaJ) * learning_rate

print(a2)
print(w2)
print(w1)
