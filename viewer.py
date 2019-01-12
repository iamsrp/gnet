#!/usr/bin/env python3
'''
Simple and very cheesy viewer. 

Adapted from:
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

# ----------------------------------------------------------------------
# Constants

MOVE_EYE   = "MOVE_EYE"
MOVE_EYE_2 = "MOVE_EYE_2"
TRANS      = "TRANS"
ZOOM       = "ZOOM"

LINK_MOD = 3

g_fViewDistance = 9.0
g_Width         = 1024
g_Height        =  768

g_nearPlane =    1.0
g_farPlane  = 1000.0

# Global variables
action = ""
xStart = 0.0
yStart = 0.0
zoom   = 65.0

xRotate = 0.0
yRotate = 0.0
zRotate = 0.0

xTrans = 0.0
yTrans = 0.0

graph = None

#-------------------

def scenemodel():
    glRotate(-90, 0.0, 0.0, 1.0)

    # The shape
    num_layers = graph.num_layers
    x_offset   = (1-num_layers) / 2
    (x_scale, y_scale, z_scale) = (2, 0.25, 0.25)

    node2pos = dict()

    layers = [[] for i in range(num_layers)]
    for node in graph.nodes:
        layers[node.depth].append(node)

    # Determine the largest "count" value, for scaling the grids
    max_count = 0
    for nodes in layers:
        max_count = max(max_count, int(math.sqrt(len(nodes)) + 1))

    # Now position and render the nodes
    for nodes in layers:
        count = int(math.sqrt(len(nodes)) + 1)
        y_offset = -count / 2
        z_offset = y_offset
        for z in range(count):
            for y in range(count):
                pos = z * count + y
                if pos >= len(nodes):
                    break
                node = nodes[pos]
                x = node.depth

                count_scale = math.sqrt(max_count / count)

                # Remember the position
                pos = (float(x + x_offset) * x_scale,
                       float(y + y_offset) * y_scale * count_scale,
                       float(z + z_offset) * z_scale * count_scale)

                node2pos[node] = pos

                # What sort of node?
                if node.node_type == NodeType.IN:
                    color = [1.0, 0.0, 0.0 ,1.0]
                elif node.node_type == NodeType.MID:
                    color = [0.0, 1.0, 0.0 ,1.0]
                elif node.node_type == NodeType.OUT:
                    color = [0.0, 0.0, 1.0 ,1.0]
                else:
                    # Should not happen
                    color = [1.0, 1.0, 1.0 ,1.0]

                # Position it
                glPushMatrix()
                glTranslatef(pos[0], pos[1], pos[2])

                # Render it
                glMaterialfv(GL_FRONT, GL_DIFFUSE, color)
                glutSolidSphere(0.1, 20, 20)

                # And done
                glPopMatrix()


        # Now draw the connections
        count = 0
        for (node, pos) in node2pos.items():
            for referee in node.referees:
                if node in node2pos and referee in node2pos:
                    skip = LINK_MOD >= 1 and (count % LINK_MOD) > 0
                    count += 1
                    if skip:
                        continue
                    glPushMatrix()
                    glBegin(GL_LINES)
                    glMaterialfv(GL_FRONT_AND_BACK,
                                 GL_AMBIENT,
                                 [1.0, 1.0, 1.0, 0.2])
                    glVertex3fv(node2pos[referee])
                    glVertex3fv(node2pos[node   ])
                    glEnd()
                    glPopMatrix()


def print_help(): 
    print( """    
-------------------------------------------------------------------
Left Mousebutton       - move eye position (+ Shift for third axis)
Middle Mousebutton     - translate the scene
Right Mousebutton      - move up / down to zoom in / out
<R> Key                - reset viewpoint
<Q> Key                - exit the program
-------------------------------------------------------------------
""")


def init():
    glEnable(GL_NORMALIZE)
    glLightfv(GL_LIGHT0,GL_POSITION,[ .0, 10.0, 10., 0. ] )
    glLightfv(GL_LIGHT0,GL_AMBIENT,[ .0, .0, .0, 1.0 ]);
    glLightfv(GL_LIGHT0,GL_DIFFUSE,[ 1.0, 1.0, 1.0, 1.0 ]);
    glLightfv(GL_LIGHT0,GL_SPECULAR,[ 1.0, 1.0, 1.0, 1.0 ]);
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glShadeModel(GL_SMOOTH)
    reset_view()


def reset_view():
    global zoom, xRotate, yRotate, zRotate, xTrans, yTrans
    zoom = 65.
    xRotate = 0.
    yRotate = 0.
    zRotate = 0.
    xTrans = 0.
    yTrans = 0.
    glutPostRedisplay()


def display():
    # Clear frame buffer and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    # Set up viewing transformation, looking down -Z axis
    glLoadIdentity()
    gluLookAt(0, 0, -g_fViewDistance, 0, 0, 0, -.1, 0, 0)   #-.1,0,0
    # Set perspective (also zoom)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(zoom, float(g_Width)/float(g_Height), g_nearPlane, g_farPlane)
    glMatrixMode(GL_MODELVIEW)
    # Render the scene
    polar_view()
    scenemodel()
    # Make sure changes appear onscreen
    glutSwapBuffers()


def reshape(width, height):
    global g_Width, g_Height
    g_Width = width
    g_Height = height
    glViewport(0, 0, g_Width, g_Height)
    

def polar_view():
    glTranslatef( yTrans/100., 0.0, 0.0 )
    glTranslatef(  0.0, -xTrans/100., 0.0)
    glRotatef( -zRotate, 0.0, 0.0, 1.0)
    glRotatef( -xRotate, 1.0, 0.0, 0.0)
    glRotatef( -yRotate, .0, 1.0, 0.0)
   

def keyboard(key, x, y):
    if key == b'r':
        reset_view()
        glutPostRedisplay()

    if key == b'q':
        exit(0)


def mouse(button, state, x, y):
    global action, xStart, yStart
    if button == GLUT_LEFT_BUTTON:
        if glutGetModifiers() == GLUT_ACTIVE_SHIFT:
            action = MOVE_EYE_2
        else:
            action = MOVE_EYE
    elif button == GLUT_MIDDLE_BUTTON:
        action = TRANS
    elif button == GLUT_RIGHT_BUTTON:
        action = ZOOM

    xStart = x
    yStart = y


def motion(x, y):
    global zoom, xStart, yStart, xRotate, yRotate, zRotate, xTrans, yTrans
    if action == MOVE_EYE:
        xRotate += x - xStart
        yRotate -= y - yStart
    elif action == MOVE_EYE_2:
        zRotate += y - yStart
    elif action == TRANS:
        xTrans += x - xStart
        yTrans += y - yStart
    elif action == ZOOM:
        zoom -= y - yStart
        if zoom > 150.:
            zoom = 150.
        elif zoom < 1.1:
            zoom = 1.1
    else:
        print("unknown action\n", action)

    xStart = x
    yStart = y 

    glutPostRedisplay()

# ----------------------------------------------------------------------

if __name__=="__main__":
    # Parse the options
    (optlist, args) = getopt.getopt(sys.argv[1:], '')
    if len(args) != 1:
        print("Usage: %s <net_file>")
        exit(1)

    # Read in the graph
    with open(args[0], 'r') as fh:
        json = fh.readline()
    graph = Graph.from_json(json)

    # GLUT Window Initialization
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE |
                        GLUT_RGB    |
                        GLUT_DEPTH)
    glutInitWindowSize(g_Width, g_Height) 
    glutInitWindowPosition(0 + 4, g_Height // 4)
    glutCreateWindow("Net Viewer")

    # Initialize OpenGL graphics state
    init()

    # Register callbacks
    glutReshapeFunc (reshape)
    glutDisplayFunc (display)    
    glutMouseFunc   (mouse)
    glutMotionFunc  (motion)
    glutKeyboardFunc(keyboard)

    # Tell the user how to work it
    print_help()

    # Turn the flow of control over to GLUT
    glutMainLoop()
