import numpy as np
import matplotlib.pyplot as plt


class SVM:

    def __init__(self, learning_rate=0.001,lambdaa=0.01, iter=1000):
        self.lr = learning_rate
        self.lambdaa= lambdaa
        self.iter =iter
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for it in range(self.iter):
            for index, x in enumerate(X):
                new_data = y[index] * (np.dot(x, self.w) - self.b) >= 1
                if new_data:
                    dw=(2 * self.lambdaa * self.w)
                    self.w -= self.lr * dw
                else:
                    dw=(2 * self.lambdaa * self.w - np.dot(x, y[index]))
                    self.w -= self.lr * dw
                    self.b -= self.lr * y[index]


    def predict(self, X):
        pr = np.dot(X, self.w) - self.b
        return np.sign(pr)

X = np.array([[  7.12731332, -4.4394424 ],
 [  6.68873898 , -2.44840134],
 [ -1.1004791  , -7.78436803],
 [  3.99337867  ,-4.90451269],
 [ -1.8171622  , -9.22909875],
 [ -2.05521901 ,-10.23141199],
 [  4.20397723 , -3.61164749],
 [ -0.21804625 , -9.21962706],
 [  5.19327641 , -6.38845134],
 [ -1.83682056 , -8.21952131]])
y = np.array([1 ,1 ,-1 , 1 ,-1 ,-1 , 1 ,-1 , 1 ,-1])
svm = SVM()
svm.fit(X, y)

print(svm.w, svm.b)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X[:,0], X[:,1], marker='o',c=y)

x0_1 = np.amin(X[:,0])
x0_2 = np.amax(X[:,0])

def hyperplane(x, w, b, offset):
          return (-w[0] * x + b + offset) / w[1]

x1_1 = hyperplane(x0_1, svm.w, svm.b, 0)
x1_2 = hyperplane(x0_2, svm.w, svm.b, 0)

x1_1_m = hyperplane(x0_1, svm.w, svm.b, -1)
x1_2_m = hyperplane(x0_2, svm.w, svm.b, -1)

x1_1_p = hyperplane(x0_1, svm.w, svm.b, 1)
x1_2_p = hyperplane(x0_2, svm.w, svm.b, 1)

print(x1_1_m)
ax.plot([x0_1, x0_2],[x1_1, x1_2], 'y--')
ax.plot([x0_1, x0_2],[x1_1_m, x1_2_m], 'k')
ax.plot([x0_1, x0_2],[x1_1_p, x1_2_p], 'k')

x1_min = np.amin(X[:,1])
x1_max = np.amax(X[:,1])
ax.set_ylim([x1_min-3,x1_max+3])

plt.show()
