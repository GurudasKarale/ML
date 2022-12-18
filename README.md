# Naive_bayes

Bayes theorem is one of the most important theorem in probability to calculate conditional probabilities. So, how exactly can we  apply it in machine learning. Well, to calculate the probability of data that it belongs to particular class. This theorem can be used in binary classification problem like spam detection etc.
Bayes theorem is represented by the following equation:

        P(A|B) = P(B|A) * P(A) / P(B) 
 
        P(A|B): Posterior probability.
        
        P(A): Prior probability
        
        P(B|A): Likelihood
        
        P(B): Evidence.
        
          Say A: Class
              B:Data points


1)First separate dataset according to classes.

2)Find mean and standard deviation of dataset separated by class.

3)Mean and standard deviation calculated in above step is used to calculate the probability of real value i.e P(B/A) given in the above formula. 
We assume that B is taken from some distribution, in this case we assume it to be gaussian distribution.

4)P(A) is calculated as the ratio of number of labels belonging to that class and total number of labels.
e.g 5 datapoints belong to class 0 and 5 datapoints belong to class 1 then, P(class=0)=5/10

5)Substituting P(B/A) and P(A) in bayes theorem, we get the final probabilities.


# SVM

1)Define two hyperplanes
        
        w•xi+b ≥ +1 when yi =+1 
        
        w•xi+b ≤ -1 when yi = –1
        
        Distance between two hyperplanes:
        
        |w•x+b|/||w||=1/||w||
        
        total distance=2/||w||

2)In order to maximize the margin between hyperplanes, we need to minimize ||w||, provided that there are no datapoints between two hyperplanes.

3)Above equations for hyperplanes can be combined into:
        
        yi(xi•w) ≥ 1

4)Cost function is given as follows: 

        costFunction=lambda*(||w||^2) + (1/n)summation(max(0,1-yi(w•xi-b))
        
5)This is a constraint minimization problem, and it can be solved using lagrange multiplier. 

6)Differentiating the cost function wrt weights and bias:

        if y*f(x)>1:
	      dw=2*lambda*weights
        else:
              dw=2*lambda*weights-y*x
              db=y
                              
 7)Apply gradient descent:
 
        w=w-lr*dw
        b=b-lr*db
 
 
# Decision tree

1)take whole dataset as an input to the node.

2)Iterate over all the features and generate the question.

3)Find the gini index and finally the informaton gain of question and keep track of each question and its information gain.

4)Whichever question is having the highest info gain, consider that question for further splitting.

        probability of label=label/total label count
        gini index=1- summation{(p(x=k))^2}
        info gain=previous impurity- current impurity

# K Nearest Neighbour

1)Find eucledian distance between the datapoint to be classified and all the datapoints in the dataset.

2)Let n be the number of nearest neighbours to be found out.

3)sort datapoints in dataset in non-decreasing order of their distances.

4)Select first n sorted datapoints, and keep track of its labels count.

5)Datapoint to be classified  is nearest to the label which occurs the most.
