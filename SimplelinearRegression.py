import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self,dataset):
        self.dataset=dataset

    def mean(self,x):
        return sum(x)/float(len(x))

    def variance(self,x,mean_):
        var=0
        for point in x:
            var+=((point-mean_)**2)
        return var

    def covariance(self,x,mean1,y,mean2):
        covar=0.0
        for i in range(len(x)):
            covar+=(x[i]-mean1) * (y[i]-mean2)
        return covar

    def coefficients(self,datasett):
        x=[]
        y=[]
        for row in datasett:
            x.append(row[0])
            y.append(row[1])
        mean1,mean2=self.mean(x),self.mean(y)
        coeff1=self.covariance(x,mean1,y,mean2)/self.variance(x,mean1)
        coeff2=mean2-coeff1*mean1
        return [coeff2,coeff1]


    def regression(self):
        predict=[]
        x,y=self.coefficients(dataset)
        for row in dataset:
            pre=x+(y*row[0])
            predict.append(pre)
        return predict

dataset=[[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
lr=LinearRegression(dataset)
g=lr.regression()
print(g)
plt.scatter([1,2,4,3,5],[1,3,3,2,5])
plt.plot(g,g)
plt.show()
