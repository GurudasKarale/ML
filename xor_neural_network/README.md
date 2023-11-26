# xor_neural_network



In machine learning ,everything comes down to probability , statistics  and linear algebra. What you get at the output of the fully connected network are nothing but the conditional probabilities.
To understand the rudiments of neural network i.e the mathematics behind it , we will solve the basic XOR problem. 
So, to decide the architecture of fully connected net. there is no rule of thumb, we have to decide it on the basis of trial  and error method. For this problem we will  be using 2-2-1 architecture i.e 2 neurons in the input layer, 2 in the hidden layer and since there are two classes(1,0),one neuron in the output layer.

Feed-Forward function:
It is just the linear combination of input variables!

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/feed.JPG)
       
           Wji(1):input weights which are going to be updated once we apply learning algorithm to it.
            
           Wj0(1): biases, triggering of the neuron depends on their value.
__Total number of parameters that are required to be calculated are 9,i.e 4 weights and 2 biases in the input layer and 2 weights and 1 bias in the output layer.__

Truth table of xor:

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/xor.png)
 
__Input weight matrix size__: 2x2

__input bias vector__:1x2

__output weight matrix__:1x2

__output bias__:1x1

__classes__=[0,1]

__labels__=[0,1,1,0]

Now that we have calculated the linear combinations of input variables and added bias to it , it is time to pass it through activation function to introduce the __non linearity__. Here we are using __sigmoid__ as the activation function.

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/sigmoid.JPG)

Thus, it is denoted by z, and are the hidden layer values.

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/z.JPG)
 
Similarly, we will multiply the output of hidden layer  with output weights and add bias to it, which is going to be output of last layer.  

__Backpropagation__:

Its time to move to the most important part of the network i.e backprop! and it is carried out by taking the derivative of error and propagating it back right to the input layer using CHAIN RULE!

In first iteration, we will get some values at the output corresponding to each input , which for now are trash.

Just like we did the feed forward , now is the time to move backwards, but in this case we have the trash values as the input.

1)Firstly, we will calculate the error and it is given by: 

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/error.JPG)

gradient of the error w.r.t to weights is given  by

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/errorgradient.JPG)
 
2)derivative of activation function is given by:

__f(x)(1-f(x))__

where x is the output of specific layer(i.e say output of hidden layer is [0.4,0.8] during feed forward operation , then derivative[h(aj)] =0.4(1-.04) )

3)In order to evaluate the derivatives, we need to calculate the value of __deltaJ(error at particular unit)__ for each hidden and output unit in the network.

__deltaK=(expected output-actual output)*derivative(actual output)__

4)__deltaK is propagated backwards using chain rule.__

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/chain.JPG)
 
the above equation shows that, error depends on weights only via the summed input to Jth unit.

where

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/daj.JPG)
  
and

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/chain1.JPG)
 
i.e

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/deltaJ.JPG)
 
Finally, the gradient of error with respect to weights is given by:

![Image of feed](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/den.JPG)
 
 
5) We are almost done, only thing remaining is to apply the technique to update the weights, and here we are using gradient descent ,although there are other multiple other techniques also available like rmsprop, adamGD etc to do the same.

![Image of gradient](https://github.com/GurudasKarale/ML/blob/master/xor_neural_network/img/gradient.JPG)
 
where __η > 0__ is the learning parameter.

__At each step the weight vector is moved in the direction of the greatest rate of decrease of the error function, and so this approach is known as gradient descent or steepest descent.__

#Reference:  
  Christopher Bishop,Pattern Recognition and Machine Learning(2006).  

