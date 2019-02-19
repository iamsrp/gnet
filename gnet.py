#!/usr/bin/env python3

"""
Use the biome to generate graphs and test them.
"""

from   biome      import Biome
from   mnist      import Mnist
from   log        import LOG
from   threading  import Thread, current_thread

import math
import os
import sys
import time

# ------------------------------------------------------------------------------

class Score:
    '''
    Create a score of a graph.
    '''
    def __init__(self, graph, loss, accuracy):
        '''
        @type  graph: Graph
        @param graph:
            The graph to score.
        @type  loss: float
        @param loss:
            The graph loss.
        @type  accuracy: float
        @param accuracy:
            The graph accuracy.
        '''
        self._accuracy  = float(int(accuracy * 10000)) / 10000.0
        self._loss      = float(loss)
        self._num_nodes = 0
        self._num_cxn   = 0
        self._score     = 0

        if graph is not None:
            self._num_nodes  = len(graph.nodes)
            for node in graph.nodes:
                self._num_cxn += len(node.referees)
            # We scale the accuracy by how connected the graph is. We
            # say that a less connected graph is better, even if it
            # if it winds up with slightly lower accuracy.
            self._score = (
                self._accuracy * 
                (1.0 - 0.01 * self._num_cxn / self._num_nodes ** 2) *
                (1.0 - 0.01 * math.tanh(len(graph.mid) / len(graph.inputs)))
            )


    def cmp(self, other):
        # Higher score is better
        if self._score < other._score:
            return -1
        if self._score > other._score:
            return 1

        # Fewer is better for each of these
        if self._num_nodes < other._num_nodes:
            return 1
        if self._num_nodes > other._num_nodes:
            return -1
        if self._num_cxn < other._num_cxn:
            return 1
        if self._num_cxn > other._num_cxn:
            return -1

        # Smaller loss is better
        if self._loss < other._loss:
            return 1
        if self._loss > other._loss:
            return -1

        return 0


    def __lt__(self, other):
        return self.cmp(other) < 0


    def __le__(self, other):
        return self.cmp(other) <= 0


    def __gt__(self, other):
        return self.cmp(other) > 0


    def __ge__(self, other):
        return self.cmp(other) >= 0


    def __eq__(self, other):
        return self.cmp(other) == 0


    def __ne__(self, other):
        return self.cmp(other) != 0


    def __str__(self):
        return 'Score(score=%0.4f,acc=%0.2f%%,size=%d,#cxns=%d,loss=%0.4f)' % (
            self._score,
            self._accuracy * 100,
            self._num_nodes,
            self._num_cxn,
            self._loss
        )


# ------------------------------------------------------------------------------

_EMPTY_RESULT = { 'score'    : Score(None, 0, 1),
                  'loss'     : 1.0,
                  'accuracy' : 0.0 }

# ------------------------------------------------------------------------------

def run(runner_factory,
        seed_graph,
        population=40,
        mutation_factor=0.1,
        cull_fraction=0.5,
        num_threads=20,
        best_dir=None):
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
    @type  cull_fraction: float
    @param cull_fraction:
        The amount of the biome to cull per generation, [0..1].
    @type  mutation_factor: float
    @param mutation_factor:
        The mutation factor of the biome, [0..1].
    @type  num_threads: int
    @param num_threads:
        The number of subprocesses to evaluate the nets in.
    @type  best_dir: str or None
    @param best_dir:
        The directory to write out the best network to each round.
    '''
    # Create a child thread to do the actual work
    def thread_maker(graph, results):
        # The runner method
        def target():
            try:
                LOG.info("Running in child %s", current_thread().name)
                runner = runner_factory(graph)
                (loss, accuracy) = runner.run()
                score = Score(graph, loss, accuracy)
                results[graph] = { 'score'    : score,
                                   'loss'     : loss,
                                   'accuracy' : accuracy }
            except Exception as e:
                LOG.info("Failed to run %s: %s", graph, e)
                results[graph] = _EMPTY_RESULT

        # Create the thread with the runner method, and give it back
        thread = Thread(target=target)
        thread.setDaemon(True)
        return thread

    # The biome in which all the graphs will live
    biome = Biome("biome", seed_graph, population, mutation_factor)

    # Score all the graphs. These scores should not change (much) from
    # round to round so we don't bother re-scoring a graph which we
    # have run previously. Hence we roll the scores over between
    # rounds.
    results = {}

    # Keep going _forever_...
    round = 0
    while True:
        round += 1
        LOG.info("Round %d", round)

        # Create the list of all the graphs to process; we only need to
        # do this for new graphs. We will evaluate them in a number of
        # threads.
        graphs  = [g for g in biome.graphs if g not in results]
        threads = list()

        # Keep going until we have no more graphs to evaluate or threads to reap
        LOG.info("Evaluating %d graphs", len(graphs))
        while len(graphs) > 0 or len(threads) > 0:
            # Start num_threads children running
            while len(graphs) > 0 and len(threads) < num_threads:
                # Grab the next graph to run
                thread = thread_maker(graphs[0], results)
                graphs = graphs[1:]
                threads.append(thread)
                thread.start()

            # See if we need to wait for any threads
            if len(threads) > 0:
                new_threads = []
                for thread in threads:
                    if not thread.isAlive():
                        thread.join()
                    else:
                        new_threads.append(thread)
                threads = new_threads

            # And wait
            if len(threads) > 0:
                time.sleep(0.1)

        # Print out all the info, since some will be lost with all the other
        # info being printed out
        graphs = sorted(biome.graphs,
                        key=lambda g: results.get(g, _EMPTY_RESULT)['score'])
        LOG.info("Scores:")
        for graph in graphs:
            LOG.info("%40s %s", graph, results.get(graph, _EMPTY_RESULT)['score'])

        # Save the best one, if we have a place to put it
        if len(graphs) > 0 and best_dir is not None:
            try:
                fn = os.path.join(best_dir, "best_%08d" % round)
                LOG.info("Writing best graph as %s", fn)
                with open(fn, "w") as fh:
                    fh.write(graphs[-1].to_json())
            except Exception as e:
                LOG.warning("Failed to write out graph as %s: %s", (fn, e))

        # Now move on to the next generation
        biome.step_generation(
            cull_fraction,
            dict((g, r['score']) for (g,r) in results.items())
        )

        # Now remove any graph no long in the biome from the results
        for graph in tuple(results.keys()):
            if graph not in biome.graphs:
                del results[graph]

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    run(Mnist,
        Mnist.create_graph('seed_graph', num_mid=100),
        best_dir='/var/tmp')
