# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import numpy as np
from sys import getsizeof
from threading import Thread
import time


def compile_vertex_shader(source):
    """Compile a vertex shader from source."""
    vertex_shader = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vertex_shader, source)
    gl.glCompileShader(vertex_shader)
    # check compilation error
    result = gl.glGetShaderiv(vertex_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(vertex_shader))
    return vertex_shader


def compile_fragment_shader(source):
    """Compile a fragment shader from source."""
    fragment_shader = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fragment_shader, source)
    gl.glCompileShader(fragment_shader)
    # check compilation error
    result = gl.glGetShaderiv(fragment_shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(fragment_shader))
    return fragment_shader


def link_shader_program(vertex_shader, fragment_shader):
    """Create a shader program with from compiled shaders."""
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vertex_shader)
    gl.glAttachShader(program, fragment_shader)
    gl.glLinkProgram(program)
    # check linking error
    result = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program


# Vertex shader
VS = """
#version 330
layout (location=0) in vec2 position;
layout (location=1) in int charge;
layout (location=2) in int mass;
out int out_charge;

void main(){
    gl_PointSize = 5 * (log(mass) + 2);
    gl_Position = vec4(2*position-1, 0.5, 1.0);
    out_charge = charge;
}

"""


# Fragment shader
FS = """
#version 330
flat in int out_charge;
out vec3 color;

// Main fragment shader function.
void main()
{
    if (out_charge>=0) {color = vec3(1.0, 0.2, 0.2);}
    else {color = vec3(0.2, 0.2, 1.);}
}
"""

class Uniform(object):
    def __init__(self, glSetFunc, program, name, default=0):
        self.setter = glSetFunc
        self.location = gl.glGetUniformLocation(program, name)
        self.value = default
        self.update()


    def update(self, value=None):
        if value:
            self.value = value
        if isinstance(self.value, (list, tuple)):
            self.setter(self.location, *self.value)
        else:
            self.setter(self.location, self.value)


class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 600, 600
    # Empty objects.
    data = None
    vbo = None


    def __init__(self):
        QGLWidget.__init__(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.count = 0
        self.makeData()
        self.idleTimer = QtCore.QTimer()
        self.idleTimer.timeout.connect(self.iterate)
        self.idleTimer.start(0)
        #self.displayTimer = QtCore.QTimer()
        #self.displayTimer.timeout.connect(self.updateGL)
        #self.displayTimer.start(500)

    def iterate(self):
        damping = 0.01
        dt = 0.0005
        vs = self.velocities
        offsets = np.dstack((np.subtract.outer(self.positions[:,i],self.positions[:,i]) for i in range(self.positions.shape[-1])))
        altOffsets = np.choose(offsets < 0, (offsets-1, 1+offsets))
        offsets = np.choose(np.abs(offsets) <= np.abs(altOffsets), (altOffsets, offsets))
        distances = np.sqrt(np.sum(np.power(offsets, 2), axis=-1))
        directions = np.divide(offsets, distances[...,None])
        #forces = (interactions[...,None]) * directions * (np.reciprocal(np.power(distances,2)[...,None]))
        forces = reduce(np.multiply, (self.interactions[...,None], np.reciprocal(np.power(distances,2))[...,None], directions))
        netForces = np.sum(np.nan_to_num(forces), axis=1).clip(-1e3, 1e3)
        for i, r in enumerate(self.positions):
            self.positions[i] = (r + (dt * vs[i])) % 1
            self.velocities[i] = (1. - damping) * (self.velocities[i] + dt * netForces[i] / self.masses[i])
        
        self.count += 1
        if self.count > 2:
            self.count = 0
            self.updateGL()


    def makeData(self):
        self.N = N = 256
        #self.positions = np.array(np.random.random( (N,2)), dtype=np.float32)
        self.positions = np.array(np.random.random((self.N, 2)), dtype=np.float32)
        self.velocities = np.zeros(self.positions.shape)
        self.masses = np.ones(N, dtype=np.int)
        self.masses[:] = 10
        self.masses[-1] = 1
        #self.charges = np.array([1 if np.random.randint(2) else -1 for i in range(N)], dtype=np.int)
        self.charges = np.ones(N, dtype=np.int)
        self.charges[-1] = -1
        self.interactions = np.outer(self.charges, self.charges)


    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        self.vbo = glvbo.VBO(self.positions)
        # background color
        gl.glClearColor(.7, .7, .7, 0)
        # Allocate and assign a Vertex Array Object
        #self.vao = gl.GLuint(1)
        self.vao = gl.glGenVertexArrays(1)
        # Bind our Vertex Array Object as the current used object */
        gl.glBindVertexArray(self.vao)
        # Allocate and assign two Vertex Buffer Objects to our handle */
        vbo = gl.glGenBuffers(3)
        self.vbos = {'position': vbo[0],
                     'charge': vbo[1],
                     'mass': vbo[2],
                     }
        # Bind positions.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbos['position'])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 2*self.N*getsizeof(np.float32), self.positions, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        # Bind charges.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbos['charge'])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(self.charges)*getsizeof(np.int), self.charges, gl.GL_STATIC_DRAW)
        gl.glVertexAttribIPointer(1, 1, gl.GL_INT, 0, None)
        # Bind masses.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbos['mass'])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(self.masses)*getsizeof(np.int), self.masses, gl.GL_STATIC_DRAW)
        gl.glVertexAttribIPointer(2, 1, gl.GL_INT, 0, None)

        gl.glEnableVertexAttribArray(0);
        gl.glEnableVertexAttribArray(1);
        gl.glEnableVertexAttribArray(2);

        # compile the vertex shader
        vs = compile_vertex_shader(VS)
        # compile the fragment shader
        fs = compile_fragment_shader(FS)
        # compile the vertex shader
        self.shaders_program = link_shader_program(vs, fs)
        # Bind the program so we can set initial parameters.
        gl.glUseProgram(self.shaders_program)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)


    def keyPressEvent(self, event):
        key = event.key()
        handled = True
        if key in (QtCore.Qt.Key_Plus,):
            pass
        else:
            handled = False

        if handled:
            self.updateGL()
        else:
            event.ignore()


    def mousePressEvent(self, event):
        self.lastClickPos = event.pos()


    def mouseReleaseEvent(self, event):
        self.updateGL()


    def paintGL(self):
        """Paint the scene.
        """
        #self.positions = np.array(np.random.random((self.N, 2)), dtype=np.float32)
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # bind the VAO
        gl.glBindVertexArray(self.vao)

        # Update positions.
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbos['position'])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 2*self.N*getsizeof(np.float32), self.positions, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        # Use our pipeline.
        gl.glUseProgram(self.shaders_program)

        gl.glDrawArrays(gl.GL_POINTS, 0, self.N)


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)
        # set orthographic projection (2D only)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, 1, 1, 0, -1, 1)


if __name__ == '__main__':
    import sys
    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            # initialize the GL widget
            self.widget = GLPlotWidget()
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()


    # create the Qt App and window
    app = QtGui.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()