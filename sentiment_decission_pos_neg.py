import tensorflow as tf
import numpy as np

'''
We use the same network as for the MNIST dataset 

The changes are just marginal. The main part is preparing the data

We will see, that the data set is way to small in order to come up with a nice accuracy rate

In the YouTube example, he got 58% correct. With some playing around (5min), I managed 62% by 
taking a 
smaller batch_size of 50 instead of 100 
 
12 instead of 10 epoches


'''

### Get the data, therefore we wrote a functions which takes pos.txt, neg.txt examples and
### Prepare them in order to have training and test data
from utilities import create_feature_sets_and_labels
train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
   


n_nodes_hl1 = 500  # hl = hidden layer 1-3
n_nodes_hl2 = 500
n_nodes_hl3 = 500 

n_classes = 2      # 2 output classes pos/neg

batch_size = 50    #  datalines at a time go through our network


#Placeholder for input (if you don't fit the right stuff, it gives an error
x = tf.placeholder('float', [None,len(train_x[0])])  
y = tf.placeholder('float')

def neural_network_model(data):

    # MODELLING OF LAYERS    

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

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
    prediction = neural_network_model(x)  # output is one HOT-Array of length 2

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits =  prediction,labels =  y))
    
   
    optimizer = tf.train.AdamOptimizer().minimize(cost)  # learning rate = .0001 is default

    
    # cycles of (feed forward and back prob) (hm - how many)
    hm_epochs = 20
 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        # TRAINING

        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            ## MAIN CHANGES IN THIS PART IN ORDER TO FEED THE CORRECT BATCHES
            i = 0
            while i < len(train_x):
                start = i
                end  = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start: end])
            
                _,c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss: ', epoch_loss)


        # TEST
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))   # checking if correct

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ', accuracy.eval({x: test_x, y: test_y}))




train_neural_network(x)






