"""
A graph representation of a Neural Net.
"""

from enum import Enum

class NodeType(Enum):
    '''
    The different node types.
    '''
    IN  = 1 # An input node
    OUT = 2 # An output node
    MID = 3 # An inner node


class Node:
    '''
    A node in the graph.
    '''
    def __init__(self,
                 node_type=NodeType.MID,
                 multipler=None,
                 bias=None):
        '''
        @type  node_type: NodeType
        @param node_type:
            The type of this node.
        @type  multipler: float
        @param multipler:
            If this node's multiplier is a constant one then this is that value.
            Else it is None and, hence, variable.
        @type  bias: float
        @param bias:
            If this node's bias is a constant one then this is that value. Else
            it is None and, hence, variable.
        '''
        # Save the args
        self._type = node_type
        self._mult = float(multipler) if multipler is not None else None
        self._bias = float(bias)      if bias      is not None else None
        
        # The set of nodes to which this node refers
        self._referees = set()

        # The set of nodes which refer to this node
        self._referrers = set()

        # The transitive closure of the references, chasing all the way through
        # nodes not directly refererred to by this one. If None then it needs to
        # be computed.
        self._closure = None

        # The cache of this node's depth in the graph, this is the distance from
        # this node to the inputs
        if self._type is NodeType.IN:
            # If we are at the input layer then our depth is zero
            self._depth = 0
        else:
            # Computed later
            self._depth = None


    @property
    def referees(self):
        '''
        The set of nodes to which this node refers.

        @rtype: frozenset
        @return:
            The set of nodes which this node refers to. 
        '''
        return frozenset(self._referees)


    @property
    def node_type(self):
        '''
        The type of this code.

        @rtype: NodeType
        '''
        return self._type


    @property
    def depth(self):
        '''
        Get the depth of this node in the graph. This is the maximum distance from
        this node to the input nodes.

        @rtype: int
        @return:
            The node depth
        '''
        # Need to compute this?
        if self._depth is None:
            # We're 1 deeper than our deepest referee
            if len(self._referees) == 0:
                self._depth = 0
            else:
                self._depth = max([r.depth for r in self._referees]) + 1
        return self._depth


    @property
    def multipler(self):
        '''
        This node's multiplier, if it is a constant, else None.
        '''
        return self._mult


    @property
    def bias(self):
        '''
        This node's bias, if it is a constant, else None.
        '''
        return self._bias


    def add_referee(self, node):
        '''
        Add the given node as one which this refers. This may not be an output node.

        This will raise a L{ValueError} if:
          o The value is None
          o The given node is an output node
          o This node is an input node
          o It would cause a loop in the graph

        @type  node: None
        @param node:
            The node to add.
        '''
        # Validity checks
        if self.node_type is NodeType.IN:
            raise ValueError("An input node may not have referrees")
        if node is None:
            raise ValueError("Given node was None")
        if node.node_type is NodeType.OUT:
            raise ValueError("Given node was an output node")
        if node is self or self in node._get_closure():
            # That node refers to this one in some way, this would create a loop
            # in the graph
            raise ValueError("Adding this node would create a loop")

        # Add to the links, in both directions
        self._referees.add(node)
        node._referrers.add(self)

        # Update our closure accordingly
        self._update_closure(node)

        # And flush the depth cache
        self._flush_depth()


    def remove(self):
        '''
        Remove this node from the graph.
        
        @raises: L{ValueError} if this node was not an inner one.
        '''
        # This is done by stiching the nodes which refer to this one directly to
        # the nodes to which this one refers.
        #
        # Before:               After:
        #          O       O            O - - - O
        #            \   /                \   /  
        #              O                    X    
        #            /   \                /   \  
        #          O       O            O - - - O

        # We may not remove input or output nodes. That would be bad.
        if self.node_type is not NodeType.MID:
            raise ValueError(
                "Attempt to remove node of type %s", self.node_type
            )

        # We need to remove ourselves from the referees and the referrers, and
        # the closures, of our peers
        for referree in self._referees:
            referree._referrers.remove(self)
        for referrer in self._referrers:
            referrer._referees.remove(self)
            referrer._flush_closure()
            referrer._flush_depth()

        # Now stich the nodes together
        for referree in self._referees:
            for referrer in self._referrers:
                referrer.add_referee(referree)

        # And clear us out, in case anyone gets any bright ideas about reusing
        # this node
        self._referees .clear()
        self._referrers.clear()
        self._closure = None
        self._depth   = None
        self._type    = None


    def connects_to(self, node):
        '''
        Whether this node connects to the given node, via referrers, in some way.

        @type  node: None
        @param node:
            The node to test.
        '''
        return node in self._get_closure()


    def _get_closure(self, visited=None):
        '''
        The canonical list of nodes which are referred to by this one, either
        directly or indirectly.

        @type  visited: set
        @param visited:
            The set of nodes visited when computing the closure.
        '''
        if self._closure is None:
            # Create the closure
            self._closure = set(self._referees)

            # Now add in the closures of all the referrees
            if visited is None:
                visited = set()
                visited.add(self)
            for referree in self._referees:
                self._closure.update(referree._get_closure(visited=visited))

        # This should exist now
        return self._closure


    def _update_closure(self, referee):
        '''
        Update the closure of this Node from that of a referee child.

        @type  referee: None
        @param referee:
            The node to update our closure from.
        '''
        # If the child is not a referee of us then something is wrong.
        if referee not in self._referees:
            raise ValueError(
                "Given node was not a referee of this one"
            )

        # Okay, update ourselves. We ensure that the referee is in our closure,
        # and that all of its closure is added to ours. Remember the size before
        # we do this, so that we can see if the closure changes or not.
        closure = self._get_closure()
        size = len(closure)
        closure.add(referee)
        closure.update(referee._get_closure())

        # And now we have to update all the nodes which refer to us. We only
        # need to do this if we mutated our closure above. If we didn't change
        # then nor will our referrers.
        if len(closure) > size:
            for referrer in self._referrers:
                referrer._update_closure(self)


    def _flush_depth(self):
        '''
        Flush our depth cache, and that of our referers (i.e. upstream).
        '''
        self._depth = None
        for referrer in self._referrers:
            referrer._flush_depth()


    def _flush_closure(self):
        '''
        Flush our closure cache, and that of our referers (i.e. upstream).
        '''
        self._closure = None
        for referrer in self._referrers:
            referrer._flush_closure()


class Graph:
    '''
    A collection of L{Node}s. The graph is really defined by the Nodes and their
    connections. The graph itself doesn't keep track of the toplogy of the
    nodes, it's just responsible for higher level operations.
    '''
    def __init__(self, name, inputs, outputs):
        '''
        @type  name: str
        @param name:
            The name of this graph. Must be unique for the entire scope of this
            application's runtime.
        @type  inputs: iterable(Node)
        @param inputs:
            The input nodes in the graph. Must not be empty.
        @type  inputs: iterable(Node)
        @param outputs:
            The output nodes in the graph. Must not be empty.

        '''
        # Sanity
        if name is None:
            raise ValueError("name was None")
        if inputs is None:
            raise ValueError("inputs was None")
        if outputs is None:
            raise ValueError("outputs was None")

        # Save our name
        self._name = name

        # All the nodes in the graph, broken up by type
        self._inputs  = tuple(inputs)
        self._outputs = tuple(outputs)
        self._inner   = set()

        # Sanity
        if len(self._inputs) == 0:
            raise ValueError("Given empty inputs")
        if len(self._outputs) == 0:
            raise ValueError("Given empty outputs")
        if len([n for n in self._inputs if n.node_type is not NodeType.IN]) > 0:
            raise ValueError("Input nodes were not all of input type")
        if len([n for n in self._outputs if n.node_type is not NodeType.OUT]) > 0:
            raise ValueError("Output nodes were not all of output type")

        
    @property
    def name(self):
        '''
        The name of this graph.
        '''
        return self._name


    @property
    def inputs(self):
        '''
        Get the input nodes.

        @rtype: tuple
        @return:
            The inputs.
        '''
        return self._inputs


    @property
    def outputs(self):
        '''
        Get the output nodes.

        @rtype: tuple
        @return:
            The outputs.
        '''
        return self._outputs


    @property
    def inner(self):
        '''
        Get the inner nodes.

        @rtype: set
        @return:
            The inner nodes of the graph.
        '''
        return self._inner


    @property
    def nodes(self):
        '''
        Get all the nodes in this graph. This is potentially expensive to call.
        '''
        return set(self._inputs).union(self._outputs).union(self._inner)


    @property
    def num_layers(self):
        '''
        The number of layers in this graph. This will raise an exception if the
        graph is not fully connected.

        @rtype: int
        @return: 
            The number of layers, where the input layer is zero and
            the output layer is the number of layers, minus one.
        '''
        # This will be the determined by the maximum depth of any
        # output node
        max_depth = 0

        # Check all the outputs
        for o in self.outputs:
            # Determine that it's connected
            connected = False
            for i in self.inputs:
                connected = o.connects_to(i)
                if connected:
                    break
            if not connected:
                raise ValueError(
                    "Output node %r is not connected to any input node",
                    o
                )

            # It's okay to use the depth of this one
            max_depth = max(max_depth, o.depth)

        # And the number of layers is the maximum depth plus one
        return max_depth + 1


    def add_node(self, node, referees=None):
        '''
        Add a node to the graph.

        @type  node: Node
        @param node:
            The node to add.
        @type  referees: iterable(Node), or None
        @param referees:
            The nodes to which the node will refer.
        '''
        # Sanity checks
        if self.contains_node(node):
            raise ValueError("Node already exists in graph")
        if referees is not None:
            for referee in referees:
                if not self.contains_node(referee):
                    raise ValueError("Referee not in graph: %r" % referee)

        # Okay to add then, put it in the right place
        if   node.node_type is NodeType.IN:
            self._inputs.add(node)
        elif node.node_type is NodeType.MID:
            self._inner.add(node)
        elif node.node_type is NodeType.OUT:
            self._outputs.add(node)
        else:
            raise ValueError("Unknown node type: %s", node.node_type())

        # Adding the referees to this node should always succeed
        if referees is not None:
            for referee in referees:
                node.add_referee(referee)


    def remove_node(self, node):
        '''
        Remove a node from the graph.

        @type  node: Node
        @param node:
            The node to add.
        '''
        # Sanity checks
        if   node.node_type is NodeType.IN:
            nodes = self._inputs
        elif node.node_type is NodeType.MID:
            nodes = self._inner
        elif node.node_type is NodeType.OUT:
            nodes = self._outputs

        # Check that we have the node
        if node not in nodes:
            raise ValueError("Node not in graph: %r", node)

        # Okay to remove then
        node.remove()
        nodes.remove(node)


    def is_connected(self):
        '''
        Whether the graph is sufficiently connected. This means that all the output
        nodes musty be connected to at least one input node.
        '''
        for o in self.outputs:
            connected = False
            for i in self.inputs:
                connected = o.connects_to(i)
                if connected:
                    break
            if not connected:
                return False
        return True
        

    def contains_node(self, node):
        '''
        Whether this graph contains the given node.
        '''
        return (node in self._inputs or
                node in self._inner  or
                node in self._outputs)
