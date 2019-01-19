#!/usr/bin/env python3
'''
Simple graph viewer.

Very early template code from here:
  http://carloluchessa.blogspot.com/2012/09/simple-viewer-in-pyopengl.html
and other sources.
'''

from   OpenGL.GL   import *
from   OpenGL.GLU  import *
from   OpenGL.GLUT import *

from   graph       import Graph, NodeType

import getopt
import math
import sys

class Viewer():
    '''
    A class to take a graph and render it for viewing.
    '''
    # Commands
    _YAW  = "YAW"
    _ROLL = "ROLL"
    _MOVE = "MOVE"
    _ZOOM = "ZOOM"

    def __init__(self, filename):
        '''
        Constructor.
        '''
        # Window settings etc
        self._width      = 1024
        self._height     =  768
        self._view_dist  =    9.0
        self._near_plane =    1.0
        self._far_plane  = 1000.0

        # The current action which we are performing, if any
        self._action = ""

        # The last coordinates we saw for x and y in an input
        self._last_x = 0.0
        self._last_y = 0.0

        # Set in _reset_view()
        self._zoom     = None
        self._x_rotate = None
        self._y_rotate = None
        self._z_rotate = None
        self._x_trans  = None
        self._y_trans  = None

        # How we render the links. The modulo value allows us to
        # display fewer of them, and we can toggle seeing the start
        # and ending layers.
        self._link_mod   = 11
        self._link_froms = set()
        self._link_tos   = set()

        # Read in the graph
        with open(filename, 'r') as fh:
            json = fh.readline()
        self._graph = Graph.from_json(json)

        # We see them all by default
        for i in range(self._graph.num_layers):
            self._link_froms.add(i)
            self._link_tos  .add(i)

        # Bring back to the defaults
        self._reset_view()

    #-------------------

    def _render(self):
        '''
        Render the scene from scratch.
        '''
        # The shape
        num_layers = self._graph.num_layers
        x_offset   = (1-num_layers) / 2.0
        (x_scale, y_scale, z_scale) = (2, 0.25, 0.25)

        # The positions of the nodes in space
        node2pos = dict()

        # Put the nodes into their layers
        layers = [[] for i in range(num_layers)]
        for node in self._graph.nodes:
            layers[node.depth].append(node)

        # Determine the largest values, for scaling the grids etc.
        max_count     = 0
        max_referees  = [1 for i in range(num_layers)]
        max_referrers = [1 for i in range(num_layers)]
        for nodes in layers:
            max_count = max(max_count, int(math.sqrt(len(nodes)) + 1))
            for node in nodes:
                d = node.depth
                max_referees [d] = max(max_referees [d], len(node.referees ))
                max_referrers[d] = max(max_referrers[d], len(node.referrers))

        # Now position and render the nodes
        for nodes in layers:
            count = int(math.sqrt(len(nodes)))
            y_offset = -count / 2
            z_offset = y_offset
            for pos in range(len(nodes)):
                z = int(pos / count)
                y = int(pos % count)
                node = nodes[pos]
                x = node.depth

                count_scale = math.sqrt(max_count / count)**1.25

                # Remember the position
                pos = (float(x + x_offset) * x_scale,
                       float(y + y_offset) * y_scale * count_scale,
                       float(z + z_offset) * z_scale * count_scale)

                node2pos[node] = pos

                # What sort of node?
                color = [len(node.referrers) / max_referrers[x],
                         len(node.referees)  / max_referees [x],
                         0.0,
                         1.0]

                # Position it
                glPushMatrix()
                glTranslatef(pos[0], pos[1], pos[2])

                # Render it
                glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
                glutSolidSphere(0.1, 5, 5)

                # And done
                glPopMatrix()

            # Now draw the connections
            counts = dict((n, 0) for n in node2pos.keys())
            for (node, pos) in node2pos.items():
                # Skip if this layer is not enabled as a from
                if node.depth not in self._link_froms:
                    continue

                for referee in node.referees:
                    # Skip if this layer is not enabled as a to
                    if referee.depth not in self._link_tos:
                        continue

                    # Skip if it's not a modulo-matching link
                    counts[node] += 1
                    if (counts[node] % self._link_mod) > 0:
                        continue

                    # Okay to render
                    glPushMatrix()
                    glBegin(GL_LINES)
                    glMaterialfv(GL_FRONT_AND_BACK,
                                 GL_AMBIENT,
                                 [1.0, 1.0, 1.0, 0.2])
                    glVertex3fv(node2pos[referee])
                    glVertex3fv(node2pos[node   ])
                    glEnd()
                    glPopMatrix()


    def _init(self):
        '''
        Set up the OpenGL scene how we like it.
        '''
        glEnable(GL_NORMALIZE)
        glLightfv(GL_LIGHT0,GL_POSITION,[ 0.0, 10.0, 10.0, 0.0 ] )
        glLightfv(GL_LIGHT0,GL_AMBIENT, [ 0.0,  0.0,  0.0, 1.0 ]);
        glLightfv(GL_LIGHT0,GL_DIFFUSE, [ 1.0,  1.0,  1.0, 1.0 ]);
        glLightfv(GL_LIGHT0,GL_SPECULAR,[ 1.0,  1.0,  1.0, 1.0 ]);
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glShadeModel(GL_SMOOTH)


    def _reset_view(self):
        '''
        Put the view back to a sensible place.
        '''
        self._zoom     = 65.0
        self._x_rotate =  0.0
        self._y_rotate =  0.0
        self._z_rotate =  0.0
        self._x_trans  =  0.0
        self._y_trans  =  0.0


    def _display(self):
        '''
        Do the job of setting things up to be rendered.
        '''
        # Clear frame buffer and depth buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up viewing transformation, looking down -Z axis
        glLoadIdentity()
        gluLookAt( 0.0, 0.0, -self._view_dist,
                   0.0, 0.0, 0.0,
                  -0.1, 0.0, 0.0)

        # Set perspective and zoom
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self._zoom,
                       float(self._width) / float(self._height),
                       self._near_plane, self._far_plane)

        # Put things into the right place
        glMatrixMode(GL_MODELVIEW)
        glTranslatef( self._y_trans/100.,                  0.0, 0.0)
        glTranslatef(                0.0, -self._x_trans/100.0, 0.0)

        glRotatef( -self._z_rotate, 0.0, 0.0, 1.0)
        glRotatef( -self._x_rotate, 1.0, 0.0, 0.0)
        glRotatef( -self._y_rotate, 0.0, 1.0, 0.0)

        # Rotate so that the layers are columnar
        glRotate(-90, 0.0, 0.0, 1.0)

        # Render the scene
        self._render()

        # Make sure changes appear onscreen
        glutSwapBuffers()


    def _reshape(self, width, height):
        '''
        Handle a window resize.
        '''
        self._width  = width
        self._height = height
        glViewport(0, 0, self._width, self._height)


    def _keyboard(self, key, x, y):
        '''
        Handle a user key-press.
        '''
        # Set if we want to redraw
        redraw = False

        if key == b'r':
            # 'r' pressed, for a reset
            print("Resetting view")
            self._reset_view()

        elif key == b'q' or key == b'\x1b':
            # 'q' or esc pressed
            exit(0)

        elif key == b'+' or key == b'=':
            # Want more links, so smaller mod
            self._link_mod = max(self._link_mod - 1, 1)
            print("Link mod now %d" % (self._link_mod,))
            redraw = True

        elif key == b'-':
            # Want fewer links, so bigger mod
            self._link_mod += 1
            print("Link mod now %d" % (self._link_mod,))
            redraw = True

        # Handle the user pressing 1~0, or shift-1~0, to toggle
        # rendering of the ends of the links.
        elif key in b'1234567890!@#$%^&*()':
            index = b'1234567890!@#$%^&*()'.find(key)
            which = index % 10
            if index < 10:
                name = "froms"
                set_ = self._link_froms
            else:
                name = "tos"
                set_= self._link_tos

            if which in set_:
                print("Disabling %d in %s" % (which, name))
                set_.remove(which)
            else:
                print("Enabling %d in %s" % (which, name))
                set_.add(which)

            redraw = True

        # Need to redraw?
        if redraw:
            glutPostRedisplay()


    def _mouse(self, button, state, x, y):
        '''
        Handle a mouse action.
        '''
        if button == GLUT_LEFT_BUTTON:
            if glutGetModifiers() == GLUT_ACTIVE_SHIFT:
                self._action = Viewer._ROLL
            else:
                self._action = Viewer._YAW
        elif button == GLUT_MIDDLE_BUTTON:
            self._action = Viewer._MOVE
        elif button == GLUT_RIGHT_BUTTON:
            self._action = Viewer._ZOOM
        elif button == 3:
            self._zoom = min(150, self._zoom * 1.1)
            glutPostRedisplay()
        elif button == 4:
            self._zoom = max(1.1, self._zoom * 0.9)
            glutPostRedisplay()

        # And remember where we got to
        self._last_x = x
        self._last_y = y


    def _motion(self, x, y):
        '''
        Handle mouse motion while we have an actyion.
        '''
        if self._action == Viewer._YAW:
            self._x_rotate += x - self._last_x
            self._y_rotate -= y - self._last_y
        elif self._action == Viewer._ROLL:
            self._z_rotate += y - self._last_y
        elif self._action == Viewer._MOVE:
            self._x_trans += x - self._last_x
            self._y_trans += y - self._last_y
        elif self._action == Viewer._ZOOM:
            self._zoom -= y - self._last_y
            if self._zoom > 150.:
                self._zoom = 150.
            elif self._zoom < 1.1:
                self._zoom = 1.1
        else:
            print("unknown action\n", self._action)

        self._last_x = x
        self._last_y = y

        glutPostRedisplay()


    def run(self):
        '''
        Main loop.
        '''
        # GLUT Window Initialization
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE |
                            GLUT_RGB    |
                            GLUT_DEPTH)
        glutInitWindowSize(self._width, self._height)
        glutInitWindowPosition(0 + 4, self._height // 4)
        glutCreateWindow("Net Viewer")

        # Initialize OpenGL graphics state
        self._init()

        # Register callbacks
        glutReshapeFunc (self._reshape)
        glutDisplayFunc (self._display)
        glutMouseFunc   (self._mouse)
        glutMotionFunc  (self._motion)
        glutKeyboardFunc(self._keyboard)

        # Turn the flow of control over to GLUT
        glutMainLoop()

    # ----------------------------------------------------------------------

if __name__=="__main__":
    # Parse the options
    (optlist, args) = getopt.getopt(sys.argv[1:], '')
    if len(args) != 1:
        print("Usage: %s <net_file>")
        exit(1)

    # Create the viewer and hand off control to it
    viewer = Viewer(args[0])
    viewer.run()
