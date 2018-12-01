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

# Hyper-parameters
epochs        =  10     # Total number of training epochs
batch_size    = 100     # Training batch size
learning_rate =   0.001 # The optimization initial learning rate

# MNIST config
img_h = img_w = 28             # MNIST images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
n_classes = 10                 # Number of classes, one class per digit

# Load MNIST data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Convert from 28x28 @ 255 greyscale to 784 @ 1.0
x_train = x_train.reshape([-1, img_size_flat]) / 255.0
x_test  = x_test .reshape([-1, img_size_flat]) / 255.0
# And the labels from indices to one-hot arrays
y_train = np.eye(n_classes)[y_train]
y_test  = np.eye(n_classes)[y_test]

# What do we have
print("Size of:")
print("- Training-set: %s" % str(y_train.shape))
print('x_train: %s' % str(x_train.shape))
print('y_train: %s' % str(y_train.shape))

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
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out_tensor),
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
    print('Training epoch: %d' % (epoch + 1))

    # Randomly shuffle the training data at the beginning of each epoch
    permutation = np.random.permutation(y_train.shape[0]).astype(np.int32)
    x_train = x_train[permutation, :]
    y_train = y_train[permutation]

    pos = 0
    for iteration in range(num_tr_iter):
        # Slice out this batch
        x_batch = x_train[pos : pos + batch_size]
        y_batch = y_train[pos : pos + batch_size]
        pos    += batch_size

        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % 100 == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)

            print("iter %03d:\t Loss=%.2f,\tTraining Accuracy=%.1f%%" %
                  (iteration, loss_batch, acc_batch * 100))


# ----------------------------------------------------------------------

# Test the network after training
# Accuracy
feed_dict_test = {x: x_test, y: y_test}
loss_test, acc_test = sess.run([loss, accuracy], feed_dict=feed_dict_test)
print('---------------------------------------------------------')
print("Test loss: %.2f  Test accuracy: %.1f%%" % (loss_test, acc_test * 100))
print('---------------------------------------------------------')    

# ----------------------------------------------------------------------

sess.close()
