import numpy as np
from math import sqrt,exp,pi

class nb:

    def __init__(self,dataset):                 #initialize the dataset
        self.dataset=dataset

    def separate(self):

        sept=dict()                              #initialize dictionary
        for i in range(len(dataset)):
            row=dataset[i]                   #take the ith row vector
            label=row[-1]                          #take last element of row vector which is a binary label(0,1)
            if(label not in sept):
                sept[label]=list()                 #store labels in the dictionary
            sept[label].append(row)                #store row vectors corresponding to labels in the dictionary

        return sept

    def mean(self,columnVector):
        return sum(columnVector)/float(len(columnVector))

    def stdDeviation(self,columnVector):

        expectedValue=self.mean(columnVector)

        variance=sum([(x-expectedValue)**2 for x in columnVector]) / float(len(columnVector)-1)

        return sqrt(variance)




    def meanOfVector(self,rows):

        meann=[]
        for column in zip(*rows):
            meann.append((self.mean(column),self.stdDeviation(column),len(column)))

        del(meann[-1])
        return meann

    def finalMean(self,sp):


        saveMean=dict()
        for key, rows in sp.items():
            saveMean[key]=self.meanOfVector(rows)
        return saveMean

    def probabilityGaussian(self,x, mean, stdev):
	    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	    probabilityDensity=(1 / (sqrt(2 * pi) * stdev)) * exponent
	    return probabilityDensity

    def conditionalProbability(self,fm,row):

        count=sum([fm[p][0][2] for p in fm])
        prob=dict()
        for key,val in fm.items():
            prob[key]=fm[key][0][2]/float(count)

            for j in range(len(val)):

                mean, stdev, _ = val[j]
                prob[key] *=self.probabilityGaussian(row[j], mean, stdev)


        return prob





dataset=[[4.35432,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]

obj=nb(dataset)
sp=obj.separate()
fm=obj.finalMean(sp)
cp=obj.conditionalProbability(fm,dataset[5])

print(cp)

