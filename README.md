# naive_bayes

Bayes theorem is one of the most important theorem in probability to calculate conditional probabilities. So, how exactly can we  apply it in machine learning. Well, to calculate the probability of data that ,it belongs to particular class. This theorem can be used in binary classification problem like spam detection etc.
Bayes theorem is given by:

        P(A|B) = P(B|A) * P(A) / P(B) 
 
        P(A|B): Posterior probability.
        
        P(A): Prior probability
        
        P(B|A): Likelihood
        
        P(B): Evidence.
        
          Say A: Class
              B:Values


1)First separate dataset according to classes.

2)Find mean and standard deviation of dataset separated by class.

3)Mean and standard deviation calculated in above step is used to calculate the probability of real value i.e P(B/A) given in the above formula. 
We assume that B is taken from some distribution, in this case we assume it to be gaussian distribution.

4)P(A) is calculated as the ratio of number of labels belonging to that class and total number of labels.
e.g 5 datapoints belong to class 0 and 5 datapoints belong to class 1 then, P(class=0)=5/10

5)Substituting P(B/A) and P(A) in bayes theorem, we get the final probabilities.
