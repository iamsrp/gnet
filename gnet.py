#!/usr/bin/env python3

"""
Use the biome to generate graphs and test them.
"""

from   biome      import Biome
from   mnist      import Mnist
from   log        import LOG
from   sys        import float_info

import math
import os
import sys

# ------------------------------------------------------------------------------

def run(runner_factory,
        seed_graph,
        population=10,
        mutation_factor=0.1,
        num_processes=1):
    '''
    Starting with a seed graph, keep working to produce more networks.

    @type  runner_factory: function(Graph -> NetRunner)
    @param runner_factory:
        The factory method for making a L{NetRunner} for a given graph.
    @type  seed_graph: Graph
    @param seed_graph:
        The graph to start the biome with.
    @type  population: int
    @param population:
        The population count of the biome.
    @type  mutation_factor: float
    @param mutation_factor:
        The mutation factor of the biome, [0..1].
    @type  num_processes: int
    @param num_processes:
        The number of subprocesses to evaluate the nets in.
    '''
    # The biome in which all the graphs will live
    biome = Biome("biome", seed_graph, population, mutation_factor)

    # Score all the graphs. These scores should not change (much) from
    # round to round so we don't bother re-scoring a graph which we
    # have run previously. Hence we roll the scores over between
    # rounds.
    scores = {}

    # Keep going _forever_...
    round = 0
    while True:
        round += 1
        LOG.info("Round %d", round)

        # Create the list of all the graphs to process; we only need to
        # do this for new graphs. We will evaluate them in a number of
        # sub-processes.
        graphs = [g for g in biome.graphs if g not in scores]
        results  = list()

        # Keep going until we have no more graphs to evaluate or results to read
        LOG.info("Evaluating %d graphs", len(graphs))
        while len(graphs) > 0 or len(results) > 0:
            # Start num_processes children running
            while len(graphs) > 0 and len(results) < num_processes:
                # Grab the next graph to run
                graph  = graphs[0]
                graphs = graphs[1:]

                # We will talk over a fifo
                fifo_name = "/tmp/gnet_%d_%s" % (os.getuid(), graph.name)
                if os.path.exists(fifo_name):
                    os.unlink(fifo_name)
                os.mkfifo(fifo_name)

                # Fork a child to do the actual work
                pid = os.fork()
                if pid == 0:
                    # Child process. We open the fifo to write out the result.
                    pid = os.getpid()
                    LOG.info("Child process %d opens %s", pid, fifo_name)
                    with open(fifo_name, 'w') as fifo:
                        try:
                            LOG.info("Running in child %d", pid)
                            runner = runner_factory(graph)
                            (loss, accuracy) = runner.run()
                            score = _score(graph, loss, accuracy)
                        except Exception as e:
                            LOG.info("Failed to run %s: %s", graph, e)
                            score = -float_info.max
                        fifo.write('%f\n' % score)
                        LOG.info("Child process %d wrote score of %f to %s",
                                 pid, score, fifo_name)

                    # And now the child is done, die
                    sys.exit(0)

                else:
                    # Parent process. We need to open the fifo for
                    # reading so that the child process does not
                    # block.
                    results.append((graph, pid, open(fifo_name, 'r')))

            # See if we have something to read
            if len(results) > 0:
                (graph, pid, fifo) = results[0]
                results       = results[1:]
                score         = float(fifo.readline())
                scores[graph] = score
                fifo.close()
                LOG.info("Read score of %f from child %d", score, pid)

        # Print out all the info, since some will be lost with all the other
        # info being printed out
        LOG.info("Scores:")
        for graph in sorted(biome.graphs,
                            key=lambda g: scores.get(g, -float_info.max)):
            LOG.info("%40s %s", graph, scores.get(graph, None))

        # Now move on to the next generation, culling the bottom 50%
        biome.step_generation(0.5, scores)

        # Now remove any graph no long in the biome from the scores
        for graph in tuple(scores.keys()):
            if graph not in biome.graphs:
                del scores[graph]


def _score(graph, loss, accuracy):
    '''
    Score a graph given its loss and accuracy.

    @type  graph: Graph
    @param graph:
        The graph to score.
    @type  loss: float
    @param loss:
        The loss value from the L{NetRunner}.
    @type  accuracy: float
    @param accuracy:
        The accuracy value from the L{NetRunner}.
    '''
    return (math.sqrt((1.0 - float(loss)    )**2 +
                      (      float(accuracy))**2) /
            math.log10(100.0 + len(graph)))

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run(Mnist, Mnist.create_graph('seed_graph'))
