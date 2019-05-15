# gnet

GNet is a NN creation system which uses genetic algorithms for creating a series of neural nets. Those nets are then trained and evaluated and child nets, with mutations, created from them. The whole process then repeats, over and over again. The networks are not layered in the tradition way that a regular NN is, any node may connect to any other node, with the limitations that no cycles are allowed, and that input and output nodes have special connection semantics.

The `gnet.py` script is the entry point and will train on the MNist data. This will write out the best network to tempdir on each iteration. Currently this works with tensorflow 1.12.0, it probably won't work with other versions.

The `veiewer.py` script allows you to view the resultant networks.

This is mostly toy code, but it seems to work pretty well. A very minimal generated network (with only about 220 nodes in the hidden layers and 95K total connections) yields an accuracy of 98.2%. Well, it feels pretty good anyhow; your judgement may be better than mine.
