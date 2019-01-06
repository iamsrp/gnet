#!/usr/bin/env python3

"""
Test the gnet code.
"""

from   biome  import Biome
from   graph  import Graph, Node, NodeType
from   log    import LOG
from   net    import NetMaker
from   random import random

import sys

# ------------------------------------------------------------------------------

def assert_raises(fn, msg):
    '''
    Assert that the given function raises an exception.

    @type  fn: function () -> ()
    @param fn:
        The function to call.
    @type  msg: str
    @param msg:
        The message to show if the function does not raise.
    '''
    try:
        fn()
        assert False, msg
    except Exception as e:
        LOG.debug("OK[%s]: %s", msg, e)


# ------------------------------------------------------------------------------

def test_add_referee():
    '''
    Test the add_referee() method's invariants and so forth.
    '''
    in0  = Node(node_type=NodeType.IN)
    mid0 = Node()
    out0 = Node(node_type=NodeType.OUT)

    # All of these should start with a depth of zero. This should set
    # up the depth caches which should change later.
    assert in0 .depth == 0
    assert mid0.depth == 0
    assert out0.depth == 0

    # None of these should work
    assert_raises(
        lambda: in0.add_referee(mid0),
        "Adding a referee to an input node should fail"
    )
    assert_raises(
        lambda: mid0.add_referee(out0),
        "Adding an output node as a referee should fail"
    )
    assert_raises(
        lambda: mid0.add_referee(mid0),
        "Adding a node as a referee of itself should fail"
    )

    # Building a simple chain should work
    out0.add_referee(mid0)
    mid0.add_referee(in0 )

    # Check the depths
    assert in0 .depth == 0
    assert mid0.depth == 1
    assert out0.depth == 2

    # Now we can check the connectivity
    assert mid0 in out0.referees
    assert in0  in mid0.referees
    assert out0 in mid0._referrers
    assert mid0 in in0 ._referrers
    assert len(out0.referees) == 1
    assert len(mid0.referees) == 1
    assert len(mid0._referrers) == 1
    assert len(in0 ._referrers) == 1

    # And check closures
    assert out0.connects_to(mid0)
    assert out0.connects_to(in0)
    assert mid0.connects_to(in0)
    assert len(in0 ._get_closure()) == 0
    assert len(mid0._get_closure()) == 1
    assert len(out0._get_closure()) == 2

    # Attempt to make a loop, this should fail
    mid1 = Node()
    mid2 = Node()
    mid0.add_referee(mid1)
    mid1.add_referee(mid2)
    assert_raises(lambda: mid2.add_referee(mid0), "Adding a loop should fail")


def test_remove_node():
    '''
    Test the remove_node() method.
    '''
    # Make the nodes
    count = 3
    ins  = [Node(node_type=NodeType.IN)  for i in range(count)]
    mids = [Node()                       for i in range(count)]
    outs = [Node(node_type=NodeType.OUT) for i in range(count)]

    # Hook them up, fully-connected
    for r in ins:
        for n in mids:
            n.add_referee(r)
    for r in mids:
        for n in outs:
            n.add_referee(r)

    # Check the depths
    for (depth, nodes) in enumerate((ins, mids, outs)):
        for n in nodes:
            assert n.depth == depth

    # Now, we yank out the middle nodes, and we should have the outs
    # directly connected to the ins.
    for n in mids:
        n.remove()
    for i in ins:
        for o in outs:
            assert i in o.referees
            assert o in i._referrers

    # Check the depths again
    for (depth, nodes) in enumerate((ins, outs)):
        for n in nodes:
            assert n.depth == depth


def test_graph():
    '''
    Test the workings of the graph.
    '''
    # Make the nodes
    count = 5
    ins  = [Node(node_type=NodeType.IN)  for i in range(count)]
    mids = [Node()                       for i in range(count)]
    outs = [Node(node_type=NodeType.OUT) for i in range(count)]

    # These should fail
    assert_raises(lambda: Graph(None, None),
                  "Creating graph with Nones")
    assert_raises(lambda: Graph(None, tuple()),
                  "Creating graph with Nones and empty")
    assert_raises(lambda: Graph(tuple(), None),
                  "Creating graph with Nones and empty")
    assert_raises(lambda: Graph(tuple(), tuple()),
                  "Creating graph with empties")
    assert_raises(lambda: Graph(outs, ins),
                  "Creating graph with outs and ins")
    assert_raises(lambda: Graph(outs, outs),
                  "Creating graph with outs and outs")
    assert_raises(lambda: Graph(ins, ins),
                  "Creating graph with ins and ins")

    # Create the graph
    graph = Graph("test_graph", ins, outs)
    for n in mids:
        graph.add_node(n)

    # Attempt to add all the nodes again, this should always fail
    for n in ins + mids + outs:
        assert_raises(lambda: graph.add_node(n),
                      "Adding a node twice should fail")

    # The graph should not be connected at this point
    assert not graph.is_connected()

    # Hook everything up, fully connected
    for r in ins:
        for n in mids:
            n.add_referee(r)
    for r in mids:
        for n in outs:
            n.add_referee(r)
    
    # The graph should now be connected
    assert graph.is_connected()

    # Give back the graph
    return graph


def test_net_maker():
    '''
    Test the workings of NetMaker.
    '''
    # Make a fully-connected graph; see test_graph()
    count = 5
    ins  = [Node(node_type=NodeType.IN)  for i in range(count)]
    mids = [Node()                       for i in range(count)]
    outs = [Node(node_type=NodeType.OUT) for i in range(count)]

    graph = Graph("test_net_maker", ins, outs)
    for n in mids:
        graph.add_node(n)

    for n in mids:
        for r in ins:
            n.add_referee(r)
    for (i, n) in enumerate(outs):
        for (j, r) in enumerate(ins):
            if ((i+j) % 2) == 1:
                n.add_referee(r)
        for (j, r) in enumerate(mids):
            if ((i+j) % 2) == 1:
                n.add_referee(r)
    

    # Now, we can create the NetMaker instance
    net_maker = NetMaker(graph)

    # And create the net with it
    net_maker.make_net()


def test_biome():
    '''
    Test the biome.
    '''
    biome = Biome("biome", test_graph(), 10, 0.1)
    scores = dict()
    for graph in biome.graphs:
        scores[graph] = random()
    biome.step_generation(0.5, scores)


def test_graph_io():
    '''
    Test rendering the graph to and from a JSON string (and, hence, a dict).
    '''
    src          = test_graph()
    src_str      = str(src)
    src_json_str = src.to_json()
    dst          = Graph.from_json(src_json_str)
    dst_str      = str(dst)
    dst_json_str = dst.to_json()
    assert src_str      == dst_str,      "%s != %s" % (src_str, dst_str)
    assert src_json_str == dst_json_str, "%s != %s" % (src_json_str, dst_json_str)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    self = sys.modules[__name__]
    for name in dir(self):
        if name.startswith('test_'):
            LOG.info("Testing %s", name)
            fn = getattr(self, name)
            fn()
