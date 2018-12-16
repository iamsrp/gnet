'''
The biome, where all the nets live.
'''

from graph  import Node, NodeType
from log    import LOG
from random import choice, random

class Biome:
    '''
    The Biome is where all the nets live. 

    It's responsible for mutating them, evaluating them, and culling
    them.
    '''
    def __init__(self, name, population, mutation_factor):
        '''
        @type  name: str
        @param name:
            The name of this biome. Must be unique for the entire scope of this
            application's runtime.
        @type  population: int
        @param population:
            The size of the population of graphs.
        '''
        # Copy in values
        self._name            = str(name)
        self._population      = int(population)
        self._mutation_factor = float(mutation_factor)

        # The list of graphs
        self._graphs     = None
        self._next_id    = 0

        # Sanity
        if self._population < 1:
            raise ValueError("Bad population count: %s", population)
        if self._mutation_factor <= 0.0:
            raise ValueError("Bad mutation factor: %s", mutation_factor)


    def populate(self, seed_graph):
        '''
        Populate this biome using the given graph as a seed.

        @type  seed_graph: Graph
        @param seed_graph:
           The graph instance to seed with.
        '''
        # Start a-fresh
        self._graphs = list()

        # Stuff in the seed_graph instance, un-mutated
        self._graphs.append(seed_graph.clone(self._next_graph_name()))

        # And now keep populating with mutated graphs
        while len(self._graphs) < self._population:
            graph = seed_graph.clone(self._next_graph_name())
            self._mutate(graph, self._mutation_factor)
            self._graphs.append(graph)


    def _next_graph_name(self):
        '''
        Get the next unique graph name.
        '''
        name = '%s_%d' % (self._name, self._next_id)
        self._next_id += 1
        return name


    def _mutate(self, graph, factor):
        '''
        Mutate the given graph, by the given mutation factor.

        @type  graph: Graph
        @param graph:
            The graph to mutate, in place.
        @type  factor: float
        @param factor:
            The mutation factor, between 0.0 and 1.0.
        '''
        def const_changer(const):
            '''Waggle a constant, depending on the factor.'''
            # Whether to waggle it
            if random() > factor:
                # Nope, just give it back
                return const
            
            # Yes, but how?
            if random() < factor:
                # Toggle it
                if const is None:
                    return 2.0 * (random() - 0.5)
                else:
                    return None
            elif const is not None:
                # No, waggle it a bit
                return max(-1.0,
                           min(1.0,
                               const + factor * (random() - 0.5)))
            else:
                return const

        # Bound the factor
        factor = min(1.0, max(0.0, float(factor)))

        # Nothing to do if the factor is zero
        if factor == 0.0:
            return

        # Get a handle on all the nodes
        nodes = graph.nodes()

        # Possibly waggle the constants in the nodes
        for node in nodes:
            node.multipler = const_changer(node.multipler)
            node.bias      = const_changer(node.bias)

        # Possibly add in a new node, for each existing one
        for i in range(len(nodes)):
            if random() > factor:
                # No, move on to the next node
                continue

            # Create the new node and add it into the graph
            node = Node()
            graph.add_node(node)

            # Connect it to a fraction of the other nodes in the
            # network
            fraction = int(1 + len(nodes) * factor)
            for i in range(fraction):
                try:
                    # Connect a random node to the new one, or the new
                    # one to a random node
                    if random() < 0.5:
                        choice(nodes).add_referee(node)
                    else:
                        node.add_referee(choice(nodes))
                except Exception:
                    pass

        # Change some connections around
        for node in nodes:
            # Do anything?
            if random() > factor:
                # Not for this node
                continue

            # Remove this node?
            if node.node_type is NodeType.MID and random() < factor:
                try:
                    # Remove this node and move on
                    graph.remove_node(node)
                    continue
                except Exception:
                    pass

            # Possibly add a new referee to this node
            if node.node_type is not NodeType.OUT and random() < factor:
                try:
                    node.add_referee(choice(nodes))
                except Exception:
                    pass
