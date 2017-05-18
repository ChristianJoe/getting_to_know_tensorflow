'''
## First steps and basic concept

In order to make my first steps with tensorflow and thus in the field of deep learning, 
I used a YouTube tutorial

https://pythonprogramming.net/tensorflow-introduction-machine-learning-tutorial/

and this is the corresponding code example, pimped with my thoughts

Credits go to https://pythonprogramming.net/ 

'''

import tensorflow as tf

'''
# COMPUTATIONS GRAPH
The are two major components to write down in tensorflow (assuming you have already propper data)

The first part deal with constructing the computational graph, the neural network, the computing structure

how many income layer nodes   
- DEPENDING ON YOUR INCOME DATA 

How many hidden layer and how many 'neurons' 
- Not to many to avoid overfitting
- take enough to gain certain accuracy
- how much power does my system have

and what is your output layer supposed to look like.
- what kind of output do i have? I/0

Income AND output depend on the training data
hidden layer depend on the application and the hardware


'''

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1,x2)

print(result)   # the print commend does not print the result 30, it gives more a handle of the network object


'''
The second part, the magic one, is to feed the network with data, let it learn and then test ist

The FIRST example shows the initialization, the running and the closing of a session.
Output: 30
''' 

# CREATE A SESSION
sess = tf.Session()
# RUN A SESSION
print(sess.run(result))
# CLOSE A SESSION
sess.close()

'''
The second session is more compact

The nice thing is, that the result, the output is a regualr python object

''' 


# THIS RUNS THE SESSION
with tf.Session() as sess:
    output =  sess.run(result)
    print(output)

print(output)

print(sess.run(result))

# KEEP IN MIND EVERYTHING OF THE NEURAL NETWORK IS MODELED WITHIN THE SESSION
# RESULTS OF SESSION CAN BE USED AS PYTHON VARIABLES