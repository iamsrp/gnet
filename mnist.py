#!/usr/bin/env python3

"""
Wrapper code for using MNIST data.
"""

from   biome      import Biome
from   graph      import Graph, Node, NodeType
from   log        import LOG
from   net        import NetRunner

import numpy      as     np
import tensorflow as     tf

import sys

# ----------------------------------------------------------------------

class Mnist(NetRunner):
    '''
    The runner for Mnist data.
    '''
    # The Mnist data, loaded lazily
    _DATA = None

    # MNIST images are 28x28
    _IMG_H         = 28
    _IMG_W         = 28
    _IMG_SIZE_FLAT = _IMG_H * _IMG_W

    # Number of classes, one class per digit
    _N_CLASSES = 10

    @staticmethod
    def _load():
        '''
        Load the MNIST data. We do this only once.
        '''
        # Use the cached version
        if Mnist._DATA is not None:
            return Mnist._DATA

        # Load MNIST data
        (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Convert from 28x28 @ 255 greyscale to 784 @ 1.0
        x_train = x_train.reshape([-1, Mnist._IMG_SIZE_FLAT]) / 255.0
        x_test  = x_test .reshape([-1, Mnist._IMG_SIZE_FLAT]) / 255.0

        # And the labels from indices to one-hot arrays
        y_train = np.eye(Mnist._N_CLASSES)[y_train]
        y_test  = np.eye(Mnist._N_CLASSES)[y_test]

        # Cache
        Mnist._DATA = (x_train, y_train, x_test, y_test)

        # And return
        return Mnist._DATA


    @staticmethod
    def create_graph(name, num_mid=200):
        '''
        Create a graph instance for use with Mnist.

        @type  name: str
        @param name:
            The name of the graph.
        @type  num_mid: int
        @param num_mid:
            The number of nodes in the middle hidden layer.
        '''
        # Create the net graph, first the nodes
        ins  = [Node(node_type=NodeType.IN)  for i in range(Mnist._IMG_SIZE_FLAT)]
        mids = [Node()                       for i in range(int(num_mid))]
        outs = [Node(node_type=NodeType.OUT) for i in range(Mnist._N_CLASSES)]

        # Connect them up
        for r in ins:
            for n in mids:
                n.add_referee(r)
        for r in mids:
            for n in outs:
                n.add_referee(r)

        # Put them into the graph
        graph = Graph(name, ins, outs)
        for n in mids:
            graph.add_node(n)
        assert graph.is_connected()

        # And give it back
        return graph


    def __init__(self, graph=None):
        super(Mnist, self).__init__(
            graph if graph is not None
            else Mnist.create_graph("mnist")
        )


    def _load_data(self):
        '''
        @see: NetRunner._load_data()
        '''
        return Mnist._load()

# ----------------------------------------------------------------------

if __name__ == "__main__":
    runner = Mnist()
    result = runner.run()
    print("Result: %s" % (result,))

