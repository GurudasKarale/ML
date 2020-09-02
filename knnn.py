from math import sqrt

class Knn:

    def __init__(self,dataset,test,n):
        self.dataset=dataset
        self.test=test
        self.n=n


    def findEucleadian(self,row1,row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i]-row2[i])**2
        return sqrt(distance)

    def neighbours(self,train,test,count):
        distance=[]                                                            #list to store the row and distance
        for row in train:
            euclid=self.findEucleadian(row,test)
            distance.append([row,euclid])

        distance.sort(key=lambda tup:tup[1])                                   #sort rows in ascending order of their distances

        neighbour=[]                                                           #list to store n neighbours
        for i in range(count):
            neighbour.append(distance[i][0])
        return neighbour

    def predict(self):
        neighbors = self.neighbours(self.dataset,self.test, self.n)
        output=[]
        for row in neighbors:
            output.append(row[-1])                   #append last value of the list i.e label of the row
        prediction = max(set(output), key=output.count)
        return prediction                             #return the label with maximum count

dataset = [[2.7,2.5,0],
	[2.5,2.3,0],
	[3.3,4.4,0],
	[1.3,1.85,0],
	[4.06,3.005,0],
	[7.68,2.08,1],
	[8,1.7,1],
	[8,-0.24,1],
	[9,3.50,1]]

knn=Knn(dataset,dataset[5],3)
k=knn.predict()
print(k)
