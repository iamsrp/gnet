'''
The biome, where all the nets live.
'''

from graph  import Node, NodeType
from log    import LOG
from random import choice, random, shuffle
from sys    import float_info

import math

# ------------------------------------------------------------------------------

class Biome:
    '''
    The Biome is where all the nets live. 

    It's responsible for mutating them, evaluating them, and culling
    them.
    '''
    def __init__(self, name, seed_graph, population, mutation_factor):
        '''
        @type  name: str
        @param name:
            The name of this biome. Must be unique for the entire scope of this
            application's runtime.
        @type  seed_graph: Graph
        @param seed_graph:
           The graph instance to seed with.
        @type  population: int
        @param population:
            The size of the population of graphs.
        @type  mutation_factor: float
        @param mutation_factor:
            The amount by which to mutate the graphs when spawning new 
            ones, [0..1].
        '''
        # Copy in values
        self._name            = str(name)
        self._population      = int(population)
        self._mutation_factor = max(0.0, min(1.0, float(mutation_factor)))

        # The list of graphs
        self._graphs  = None
        self._next_id = 0

        # We need at least a few graphs in the population in order to cull
        # correctly
        if self._population < 3:
            raise ValueError("Bad population count: %s" % (population,))

        # And populate the Biome
        self._populate(seed_graph)


    @property
    def graphs(self):
        '''
        The grpahs within this Biome.
        '''
        return self._graphs


    def step_generation(self, cull_fraction, scores):
        '''
        Cull the given fraction of graphs in the biome, according to the
        graphs scroes, and repopulate with what's left.

        @type  cull_fraction: float
        @param cull_fraction:
            The fraction of graphs to cull, [0..1].
        @type  scores: dict
        @param scores:
            The mapping from graph to score, where a higher score is better.
        '''
        # Make sure that we have a biome to repopulate from
        if len(self._graphs) == 0:
            raise ValueError("Empty biome")

        # Sanitise the inputs
        cull_fraction = max(0.0, min(1.0, float(cull_fraction)))

        # Put the graphs into order of score, high to low
        ordered = sorted(self._graphs, key=lambda g: scores[g], reverse=True)

        # How much of the graphs list to copy so that we have culled the desired
        # amount
        copy_to = max(2, min(len(ordered), 
                             int((1.0 - cull_fraction) * self._population)))

        # We copy across values but sampling in such a way as to bias the top
        # (healthiest) of the current generation. We do this by figuring out the
        # power which we need to raise copyTo to so as to get the current
        # size. If we were sampling 5 from 25 then we'd get 2 as the power and
        # would sample 0, 1, 4, 9 and 16. This means that we generally try to
        # cull some healthy genomes in favour of less healthy ones; that should
        # (hopefully) reduce stagnation and getting trapped in local maximas.
        self._graphs = list()
        pow = math.log(len(ordered)) / math.log(copy_to);
        for i in range(copy_to):
            # Floor the power to the integer value
            j = int(math.pow(i, pow))
            self._graphs.append(ordered[j])

        # Now put them in a random order
        shuffle(self._graphs)

        # How much room we have for children
        space = max(0, self._population - len(self._graphs))

        # Create the children from the remaining graphs
        children = list()
        index    = 0
        while len(children) < space:
            self._add_child_graph(self._graphs[index % len(self._graphs)],
                                  children)
            index += 1

        # And add them to the population
        self._graphs.extend(children)


    def _populate(self, seed_graph):
        '''
        Populate this biome using the given graph as a seed.

        @type  seed_graph: Graph
        @param seed_graph:
           The graph instance to seed with.
        '''
        # Start a-fresh
        graphs = list()

        # Stuff in the seed_graph instance, un-mutated
        graphs.append(seed_graph.clone(self._next_graph_name()))

        # And now keep populating with mutated graphs
        while len(graphs) < self._population:
            self._add_child_graph(seed_graph, graphs)

        self._graphs = graphs


    def _add_child_graph(self, parent, graphs):
        '''
        Add a graph which is a child of the given parent.

        @type  parent: Graph
        @param parent:
            The graph to spawn the child from.
        @type  graphs: list(Graph)
        @param graphs:
            The list to put the child into.
        '''
        LOG.info("Cloning from parent: %s", parent)
        while True:
            # Create a mutated child
            child = parent.clone(self._next_graph_name())
            self._mutate(child, self._mutation_factor)

            # Remove any inner nodes which have no inputs. We keep doing this
            # since removing one inner node might drop the depth of another.
            changed = True
            max_depth = child.num_layers - 1
            while changed:
                changed = False
                for node in tuple(child.mid):
                    if node.depth == 0 or node.depth >= max_depth:
                        child.remove_node(node)
                        changed = True

            if child.is_connected():
                graphs.append(child)
                LOG.info("Adding child graph:  %s", child)
                return


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

        # Get a handle on all the nodes, as a tuple
        nodes = tuple(graph.nodes)

        # Possibly waggle the constants in the nodes
        for node in nodes:
            node.multipler = const_changer(node.multipler)
            node.bias      = const_changer(node.bias)

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
                referee = choice(nodes)
                try:
                    # Connect a random node to the new one, or the new
                    # one to a random node
                    node.add_referee(referee)
                    LOG.debug("New node %s refers to %s", node, referee) 
                except Exception as e:
                    LOG.debug("Failed to make new node %s refer to %s: %s",
                              node, referee, e)
