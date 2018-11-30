#!/usr/bin/env python3

"""
Use the gnet frame to do MNIST.

Mostl copied from:
  https://github.com/easy-tensorflow/easy-tensorflow/blob/master/3_Neural_Network/Tutorials/1_Neural_Network.ipynb
"""

from   graph import Graph, Node, NodeType
from   log   import LOG
from   net   import NetMaker

import tensorflow as tf
import numpy      as np

import sys

# ----------------------------------------------------------------------

def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
                                             mnist.validation.images, mnist.validation.labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
    return x_test, y_test


def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

# ----------------------------------------------------------------------

# Hyper-parameters
epochs = 10             # Total number of training epochs
batch_size = 100        # Training batch size
display_freq = 100      # Frequency of displaying the training results
learning_rate = 0.001   # The optimization initial learning rate

# MNIST config
img_h = img_w = 28             # MNIST images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
n_classes = 10                 # Number of classes, one class per digit

# Load MNIST data
x_train, y_train, x_valid, y_valid = load_data(mode='train')
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))

print('x_train:\t{}'.format(x_train.shape))
print('y_train:\t{}'.format(y_train.shape))
print('x_train:\t{}'.format(x_valid.shape))
print('y_valid:\t{}'.format(y_valid.shape))

# Create the net graph, first the nodes
ins  = [Node(node_type=NodeType.IN)  for i in range(img_size_flat)]
mids = [Node()                       for i in range(200)]
outs = [Node(node_type=NodeType.OUT) for i in range(n_classes)]

# Connect them up
for r in ins:
    for n in mids:
        n.add_referee(r)
for r in mids:
    for n in outs:
        n.add_referee(r)

# Put them into the graph
graph = Graph("mnist", ins, outs)
for n in mids:
    graph.add_node(n)
assert graph.is_connected()

# Now, we can create the NetMaker instance
net_maker = NetMaker(graph)

# And create the net with it
layers = net_maker.make_net()

# Grab the inputs and outputs
in_tensor  = layers[ 0]
out_tensor = layers[-1]

# Ins and outs, with tweaked names
x = in_tensor
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_tensor),
                      name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(out_tensor, 1),
                              tf.argmax(         y, 1),
                              name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                          name='accuracy')

# Network predictions
cls_prediction = tf.argmax(out_tensor, axis=1, name='predictions')

# ----------------------------------------------------------------------

# Create the op for initializing all variables
init = tf.global_variables_initializer()

# Create an interactive session (to keep the session in the other cells)
sess = tf.InteractiveSession()

# Initialize all variables
sess.run(init)

# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    # Randomly shuffle the training data at the beginning of each epoch 
    x_train, y_train = randomize(x_train, y_train)
    for iteration in range(num_tr_iter):
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))

    # Run validation after every epoch
    feed_dict_valid = {x: x_valid[:1000], y: y_valid[:1000]}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')


# ----------------------------------------------------------------------

# Test the network after training
# Accuracy
x_test, y_test = load_data(mode='test')
feed_dict_test = {x: x_test[:1000], y: y_test[:1000]}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: {0:.2f}, test accuracy: {1:.01%}".format(loss_test, acc_test))
print('---------------------------------------------------------')    

# ----------------------------------------------------------------------

sess.close()
