"""
The Neural Network code for gnet.
"""

from   graph      import Graph, Node, NodeType
from   log        import LOG
from   threading  import Lock

import numpy      as     np
import tensorflow as     tf

class NetMaker:
    '''
    A class which generate a neural network from a L{Graph}.
    '''

    # How we make the scopes unique
    _SCOPE_ID   = 0
    _SCOPE_LOCK = Lock()

    def __init__(self, graph):
        '''
        @type  graph: Graph
        @param graph:
            The graph to create the network for.
        '''
        # Sanity
        if not graph.is_connected():
            raise ValueError(
                "Graph is not fully connected"
            )

        # Now we can save the graph and the sigmoid choice
        self._graph   = graph
        self._sigmoid = tf.nn.relu # <-- hard-coded for now

        # Now let's create a list of list of nodes which represents the
        # different layers-by-depth in the graph.
        num_layers   = graph.num_layers
        self._layers = [[] for i in range(num_layers)]

        # The inputs and outputs must be fixed with the indexing which is
        # defined by the graph.
        self._layers[             0] = list(graph.inputs)
        self._layers[num_layers - 1] = list(graph.outputs)

        # Now add in the inner nodes
        for node in graph.mid:
            # Make sure tht we don't have a non-inner node for some weird
            # reason.
            depth = node.depth
            assert node.node_type is NodeType.MID
            assert 0 < depth and depth < num_layers - 1

            # Okay, safe to add
            self._layers[depth].append(node)

        # Check that no layer is empty
        for (i, layer) in enumerate(self._layers):
            if len(layer) == 0:
                raise ValueError("No nodes in layer[%d]", i)

        # And create the placeholder tensor which represent the inputs
        self._input_tensor = tf.placeholder(
            tf.float32,
            shape=[None, len(graph.inputs)],
            name='inputs'
        )

        # Say what we're doing
        LOG.info("Initted with a graph: %s", self._graph)
    

    def make_net(self):
        '''
        Create the network for the graph.

        @rtype: list<tensor>
        @return: 
            List of layers with the first being the inputs and the last 
            being the outputs.
        '''
        # Do this within a scope in order to get unique variables etc.
        with NetMaker._SCOPE_LOCK:
            scope_id = NetMaker._SCOPE_ID
            NetMaker._SCOPE_ID += 1
        with tf.variable_scope("graph_%d_%d" % (id(self._graph), scope_id)):
            net = self._make_net()
            LOG.info("Created net: %s", net)
            return net


    def _make_net(self):
        '''
        The actual make_net method.

        @rtype: list<tensor>
        @return: 
            List of layers with the first being the inputs and the last 
            being the outputs.
        '''
        # We build up the graph layer by layer. This is done by creating a
        # tensor comprised of the sub-layers which go to make up the input of
        # this layer.
        #
        # Each entry in layers will be a tuple comprising:
        #  o The tensor
        #  o The mapping from node to the index of its corrsponding element in
        #    the tensor
        layers = []

        # Walk through the layers. With each one we create the tensor which we
        # will use as inputs that given layer, from the tensors in the previous
        # layer. The first layer is the input layer, and the last one is the
        # output layer.
        for (depth, layer_nodes) in enumerate(self._layers):
            if depth == 0:
                # Start off with the first layer being the inputs
                layers.append((
                    self._input_tensor,
                    dict((n, i) for (i, n) in enumerate(layer_nodes))
                ))
                LOG.debug("Layer[%d]", depth, layers[depth])
                continue

            # Now create the tensor which corresponds to the inputs of this
            # layer. This will be the concatenation of all the tensors of which
            # have elements which this layer references.

            # Determine all the layers which contain nodes which this layer
            # references
            depths = set()
            for (to_index, node) in enumerate(layer_nodes):
                for referee in node.referees:
                    depths.add(referee.depth)

            # Now use this to create a list of offsets into the input tensor
            # vector which we will build. Also build that vector.
            input_tensors = []
            input_size    = 0
            offsets       = []
            for i in range(depth):
                # One offset per layer/depth
                offsets.append(input_size)

                # If this layer is referenced then include it
                if i in depths:
                    input_tensors.append(layers[i][0])
                    input_size += len(self._layers[i])

            # Good, now we have the input tensor defined and the offsets into
            # it. Create the actual input tensor from layer ones. The first,
            # [0], dimension is the different input values, so we concatenate
            # along the second, [1].
            input_tensor = tf.concat(input_tensors, 1, name="input")

            # Now we can figure out the mask for the matrix. This is a dict from
            # an (input,output) index tuple to a constant weight value, or None
            # of the weight is variable.
            connections = {}
            for (to_index, node) in enumerate(layer_nodes):
                for referee in node.referees:
                    # The referee's layer info
                    (referee_tensor, referee_mapping) = layers[referee.depth]

                    # Determin the offset of this node in the layer tensor and,
                    # thus, in input_tensor
                    from_index = (offsets[referee.depth] +
                                  referee_mapping[referee])

                    # Set the connectivity matrix element flag. This is done by
                    # setting the constant value.
                    connections[(from_index, to_index)] = node.multipler

            # This is the shape of the connectivity mask, multipler and zeroes
            # matrices
            shape=(input_size, len(layer_nodes))

            # The weight and bias tensors. These will be the things which will
            # be tweaked when tensorflow is fitting the model.
            weights = tf.get_variable(
                'weighs_%d' % (depth,),
                dtype=tf.float32,
                shape=(input_size, len(layer_nodes)),
                initializer=tf.truncated_normal_initializer(stddev=0.01)
            )
            bias = tf.get_variable(
                'bias_%d' % (depth,),
                dtype=tf.float32,
                initializer=tf.constant(
                    0.0,
                    shape=(len(layer_nodes),),
                    dtype=tf.float32
                )
            )

            # However, we want to mask the weights and bias off because there
            # may not be full connectivity or the values might be constants.

            # Create the multiplier mask matrix. This is a matrix of booleans
            # which restricts the connection weights to certain back elements.
            # If the weight should not be touched, because either there is not
            # back connection or because it's constant, then we want to defer to
            # a constants matrix. If there is no connection then the value in
            # that matrix will be zero.
            multiplier_mask = tf.constant(
                [
                    [
                        np.bool(
                            connections.get((from_index, to_index), None) is None
                        )
                        for to_index in range(len(layer_nodes))
                    ]
                    for from_index in range(input_size)
                ],
                name='multiplier_mask_%d' % (depth,),
                dtype=tf.bool,
                shape=shape,
                verify_shape=True
            )

            # And create the matrix of multipliers, this both the masking matrix
            # (for when there are no connections) and the constant multiplier
            # values.
            multipliers = tf.constant(
                [
                    [
                        np.float32(
                            connections.get((from_index, to_index), None) or 0.0
                        )
                        for to_index in range(len(layer_nodes))
                    ]
                    for from_index in range(input_size)
                ],
                name='multipliers_%d' % (depth,),
                dtype=tf.float32,
                shape=shape,
                verify_shape=True
            )

            # The masked weights are derived using the where() method
            masked_weights = tf.where(multiplier_mask,
                                      weights,
                                      multipliers,
                                      name="multipliers_where_%d" % (depth,))

            # Similar for the bias. Only do this if we have any constant biases
            # though (we might not have).
            if len([node.bias for node in layer_nodes if bias is not None]) > 0:
                biases = tf.constant(
                    [ np.float32(0.0 if node.bias is None else node.bias)
                      for node in layer_nodes ],
                    name='biases_%d' % (depth,),
                    dtype=tf.float32,
                    shape=shape[1:],
                    verify_shape=True
                )

                bias_mask = tf.constant(
                    [ np.bool(node.bias is None) for node in layer_nodes ],
                    name='bias_mask_%d' % (depth,),
                    dtype=tf.bool,
                    shape=shape[1:],
                    verify_shape=True
                )

                # And create the masked like with the weights
                masked_biases = tf.where(bias_mask,
                                         bias,
                                         biases,
                                         name="bias_where_%d" % (depth,))
            else:
                # No masking required
                masked_biases = bias

            # To create the layer tensor we multiply the values from the
            # from_nodes with the matrix and add in the bias.
            tensor  = tf.matmul(input_tensor, masked_weights)
            tensor += masked_biases

            # And add in the sigmoid function, but not for the last layer
            if depth < self._num_layers - 1:
                tensor = self._add_sigmoid(tensor)

            # Now, finally, remember all of this in the layers
            layers.append((
                tensor,
                dict((n, i) for (i, n) in enumerate(layer_nodes))
            ))
            LOG.debug("Layer[%d]", depth, layers[depth])

        # Okay, we have built all the layers, so we can give them back
        return [l[0] for l in layers]


    @property
    def _num_layers(self):
        '''
        The number of layers.
        '''
        return len(self._layers)


    def _add_sigmoid(self, tensor):
        '''
        Add the chosen sigmoid to the given "layer".

        @type  tensor: tensor
        @param tensor:
            The tensor representing the layer that we want to add the
            sigmoid to.
        '''
        return self._sigmoid(tensor)


class NetRunner:
    '''
    The base class for evaluating nets generated from some L{Graph}.
    '''
    def __init__(self, graph):
        '''
        @type  graph: Graph
        @param graph:
            The graph to run the network for.
        '''
        # Hyper-parameters
        self._num_epochs    =  10     # Total number of training epochs
        self._batch_size    = 100     # Training batch size
        self._learning_rate =   0.001 # The optimization initial learning rate

        LOG.info("Graph: %s", graph)
        self._graph = graph

        LOG.info("Loading data")
        (self._x_train,
         self._y_train,
         self._x_test,
         self._y_test) = self._load_data()

        # Make sure that looks like what we expect
        if len(self._x_train.shape) != 2:
            raise ValueError("Bad shape for x_train: %s", self._x_train.shape)
        if len(self._y_train.shape) != 2:
            raise ValueError("Bad shape for y_train: %s", self._y_train.shape)
        if len(self._x_test.shape) != 2:
            raise ValueError("Bad shape for x_test: %s",  self._x_test.shape)
        if len(self._y_test.shape) != 2:
            raise ValueError("Bad shape for y_test: %s",  self._y_test.shape)

        # How we are set up
        LOG.info("Init params:")
        LOG.info("  Num Epochs: %d",    self._num_epochs)
        LOG.info("  Batch Size: %d",    self._batch_size)
        LOG.info("  Rate:       %0.4f", self._learning_rate)
        LOG.info("  Graph:      %s",    self._graph)
        LOG.info("  Data:",)
        LOG.info("    x_train: %s", self._x_train.shape)
        LOG.info("    y_train: %s", self._y_train.shape)
        LOG.info("    x_test:  %s", self._x_test .shape)
        LOG.info("    y_test:  %s", self._y_test .shape)


    def run(self):
        '''
        Build the network and run it. 

        @rtype: tuple(float32, float32)
        @return:
            A tuple of: loss, accuracy. Per the test data.
        '''
        LOG.info("[%s] Doing network run", self._graph.name)
        debug = (LOG.getLevelName(LOG.getLogger().level) == 'DEBUG')
        with tf.Session(config=tf.ConfigProto(log_device_placement=debug)) as sess:
            # Create the network
            LOG.info("[%s] Creating the network", self._graph.name)
            net_maker = NetMaker(self._graph)
            layers    = net_maker.make_net()

            # Grab the inputs and outputs
            in_tensor  = layers[ 0]
            out_tensor = layers[-1]

            # The number of inputs and outputs
            num_in  = self._x_train.shape[1]
            num_out = self._y_train.shape[1]

            # Check x_* & y_* and net shapes match
            if num_in != in_tensor.shape[1]:
                raise ValueError(
                    "Number of data inputs, %d, "
                    "does not match network input count, %d, "
                    "for graph %s",
                    num_in, in_tensor.shape[1], self._graph
                )
            if num_out != out_tensor.shape[1]:
                raise ValueError(
                    "Number of data outputs, %d, "
                    "does not match network output count, %d, "
                    "for graph %s",
                    num_out, out_tensor.shape[1], self._graph
                )

            # Now make the training equipment
            truth     = tf.placeholder(tf.float32, shape=[None, num_out])
            loss      = self._create_loss     (truth, out_tensor)
            accuracy  = self._create_accuracy (truth, out_tensor)
            optimizer = self._create_optimizer(loss)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Train for this epoch
            LOG.info("[%s] Training the network", self._graph.name)
            for epoch in range(self._num_epochs):
                self._do_epoch(epoch, sess, in_tensor, truth, optimizer)

            # And give back the loss and accuracy
            LOG.info("[%s] Evaluating the network", self._graph.name)
            result = sess.run((loss, accuracy),
                              feed_dict={ in_tensor : self._x_test,
                                          truth     : self._y_test })
            LOG.info("[%s] Result: loss=%0.3f accuracy=%0.2f%%",
                     result[0],
                     100 * result[1],
                     self._graph.name)
            return result
                

    def _do_epoch(self, epoch, sess, x, y, optimizer):
        '''
        Train for an epoch.
        '''
        LOG.info("[%s] Doing epoch %d", self._graph.name, epoch)

        # Randomly shuffle the training data at the beginning of each epoch
        permutation = np.random.permutation(self._y_train.shape[0]).astype(np.int32)
        x_train = self._x_train[permutation, :]
        y_train = self._y_train[permutation]

        pos = 0
        for iteration in range(int(len(y_train) / self._batch_size)):
            # Slice out this batch
            x_batch = x_train[pos : pos + self._batch_size]
            y_batch = y_train[pos : pos + self._batch_size]
            pos    += self._batch_size

            # Run optimization op (backprop)
            sess.run(
                optimizer,
                feed_dict={ x : x_batch,
                            y : y_batch }
            )


    def _load_data(self):
        '''
        An abstract method which subclasses should implement.

        @return:
            A 4-tuple of C{x_train, y_train, x_test, y_test}.
        '''
        raise NotImplementedException("Abstract method called")


    def _create_loss(self, truth, out):
        '''
        Create the loss tensor.

        @type  truth: tensor
        @param truth:
            The training data, ground truth.
        @type  out: tensor
        @param out:
            The output layer of the network

        @rtype: tensor
        @return:
            The loss function.
        '''
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=truth,
                logits=out
            ),
            name='loss'
        )


    def _create_accuracy(self, truth, out):
        '''
        Create the accuracy tensor.

        @type  truth: tensor
        @param truth:
            The training data, ground truth.
        @type  out: tensor
        @param out:
            The output layer of the network

        @rtype: tensor
        @return:
            The accuracy function.
        '''
        return tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(out,   1),
                         tf.argmax(truth, 1)),
                tf.float32
            ),
            name='accuracy'
        )


    def _create_optimizer(self, loss):
        '''
        Create the optimizer.

        @type  loss: tensor
        @param loss:
            The loss function.

        @rtype: optimizer
        @return:
            The optimizer.
        '''
        return tf.train.AdamOptimizer(
            learning_rate=self._learning_rate,
            name='optimizer'
        ).minimize(loss)
        
