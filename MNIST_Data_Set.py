'''
The MNIST dataset contains 60000 thousend handwritten ciphers in a 28x28 pixel format with corresponding labels 0-9

http://yann.lecun.com/exdb/mnist/

The idea is to write a neural network and train it such, that it can read any handwritten number

Thanks to 
https://pythonprogramming.net/tensorflow-deep-neural-network-machine-learning-tutorial/
'''



'''
General Overview

FEED FORWART NEURAL NETWORK

input > weight > hidden layer 1 (activation function)
   > weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost OR loss function (cross entropy)

optimizer > minimize cost (AdamOptimizer, Gradient decent, AdaGrad)

backpropagation (to manipulate weights)

feed forward + back prob = EPOCH

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# read in the Mnist data set
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

'''  
EXPLAIN      			one_hot = True
# 10 classes , 0-9

0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]      just one element in an array is one=HOT
2 = [0,0,1,0,0,0,0,0,0,0]
...

This is a nice representation of the labels. Out of 10 neurons in the output layer, exactly ONE is fireing

'''

### Construction of COMPUTATION GRAPH/Neural Network with 3 hidden layer

# hl = hidden layer 1-3
n_nodes_hl1 = 500  
n_nodes_hl2 = 600
n_nodes_hl3 = 600 

n_classes = 10      # 10 output classes 

batch_size = 100    # 100 img at a time go through our network



### height x width. (28x28 = 784 make a string from matrix)

###Placeholder for input (if you don't fit the right stuff, it gives an ERROR!!!, NICE)
x = tf.placeholder('float', [None,784])   
y = tf.placeholder('float')      

### Setting up the network

def neural_network_model(data):

    # MODELLING OF LAYERS    

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}


    # INPUT * WEIGHT + BIAS     (to avoid 0, to let at least one node fire)

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])

    l1 = tf.nn.relu(l1)    # activation function
    

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    
    l2 = tf.nn.relu(l2)    # activation function


    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])

    l3 = tf.nn.relu(l3)    # activation function


    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return(output)




def train_neural_network(x):
    
    # output is one HOT-Array
    prediction = neural_network_model(x)  

    ### Define the cost function which should be minimized (Maximized) 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits =  prediction,labels =  y))
    
    ### Define how to optimize (look up which kind of optimizer exist)
    optimizer = tf.train.AdamOptimizer().minimize(cost)  
    # learning rate = .0001 is default but could be adjusted
    
    
    # cycles of feed forward and back prob (hm - how many)
    hm_epochs = 12
 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        # TRAINING
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)   # chunks through your data set
                _,c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss: ', epoch_loss)


        # TEST
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))   # checking if correct

        ### This was pretty fast explained - nice functions, how do they work?
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))




### And BOOM, just apply the function and let the magic of tensorflow happen!

train_neural_network(x)






