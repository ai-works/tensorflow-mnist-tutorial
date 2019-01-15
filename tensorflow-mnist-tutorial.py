import tensorflow as tf 

'''
inmput > weighted > hidden layer 1 > activation function > hidden l 2 
> activation function > weights > output layer 

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize the cost (adam, SGD, AdaGtad)

backropogation 

feed forward + backprop = epoch
 
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 Klassen, 0-9
'''
0 = 0
1 = 1
2 = 2
...
 
one hote encoding macht 

0 = [1,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0]

'''
# define number of nodes within hidden layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # number of data where we compute the average loss and update the weights and biases afterwards 

# x = input data = 784 pixels wide 
# matrix = height * width 
# flatten 
x = tf.placeholder('float',[None,784]) # nn expects a vector with the dimension of 784, helps to control and gives error if other dimension is thrown in the net 
y = tf.placeholder('float') 

# 1. define the architecture and the variables of our neural net 
def neural_network_model(data):
    
    # (input_data * weights) + bias 
    # 1. define what are our trained parameters 
    # define dictionary with weights and biases as keys and varibales which are my weights and biases  
    # random normal = A tensor of the specified shape filled with random normal values
    # variables weights and biases will initialized with random values that are normal distributed and updated in the training lateron 
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    # 2. what will happen to the data and the learning paramters (weights and biases )
    # (input_data * weights) + bias 
    
    # tf.add = Returns x + y element-wise.
    # matmul = matrix multiplication of weights with input data 
    # ==> 1. matrixmultiplication of data and weights from our dictionary(x*weights) and the add the biases from dict on top
    # finally put (x*w + bias) in the activation function 
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2) # l1 = die gewichteten input daten durch die aktivierungsfunktion ist der input von layer 2 

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)    

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    
    return output


## 2. define the training(how will the variables updated) of our  and training hyperparamters 

def train_neural_network(x):
    
    # 1. make predicitons
    prediction = neural_network_model(x)
    
    # 2. define cost function 
    ## softmax cross entropy = compares the prediction with the true label for one digit 
    ## reduce mean = Computes the mean of elements across dimensions of a tensor so it is the average error across our predictions on the trainingdata
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    
    # 3. define optimizer and what to minimize 
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # learningrate = 0.001
    
    # 4. define cycles of feed forward + backprop 
    hm_epochs = 10 
    
    with tf.Session() as sess:
        #sess.run(tf.initialize_all_variables())
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0 
            # underscore = variable die uns nicht interessiert braucht kein namen
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs,'loss:',epoch_loss)
            
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        

train_neural_network(x)



