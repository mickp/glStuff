# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import ctypes
import struct
DT = 0.0001
DAMPING = 0.01
NUM_PARTICLES = 4096

def compile_shader(source, shader_type):
    """Compile a shader."""
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    result = gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetShaderInfoLog(shader))
    return shader


def link_shaders(*shaders):
    """Link an arbitrary number of shaders."""
    program = gl.glCreateProgram()
    for shader in shaders:
        gl.glAttachShader(program, shader)
    gl.glLinkProgram(program)
    # check linking error
    result = gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(gl.glGetProgramInfoLog(program))
    return program


# Vertex shader
VERTEX = """
#version 330
attribute vec4 attr;
flat out float fs_charge;

void main(){
    vec2 position = vec2(attr[0], attr[1]);
    float charge = attr[2];
    float mass = attr[3];
    gl_PointSize = 5 * (log(mass) + 2);
    gl_Position = vec4(2*position-1, 0.5, 1.0);
    fs_charge = charge;
}
"""

# Fragment shader
FRAGMENT = """
#version 330
flat in float fs_charge;

void main(){
    if (fs_charge > 0){
        gl_FragColor = vec4(1., 0., 0., 1.);
    } else if (fs_charge < 0) {
        gl_FragColor = vec4(0., 0., 1., 1.);
    } else {
        gl_FragColor = vec4(0., 0., 0., 1.);
    }
}
"""

# Compute shader
COMPUTE = """
#version 330
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout( std430, binding=0 ) buffer Attributes {
    vec4 attributes[%d];
};

layout( std430, binding=1 ) buffer Velocities {
    vec2 velocities[%d];
};

uniform uint N;
uniform float dt;
uniform float damping;

layout( local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

vec2 shortest(vec2 p) {
    int i;
    for (i=0; i <=1; i++) {
        if ( abs(p[i] + 1) < abs(p[i]) ) {
            p[i] += 1;
        }
        else if (abs(p[i] - 1) < abs(p[i])) {
            p[i] -= 1;
        }
    }
    return p;
};



vec2 bound(vec2 p) {
    int i;
    for (i=0; i<=1; i++) {
        if (p[i] > 1)
        {
            p[i] -= 1;
        }
        else if (p[i] < 0) 
        {
            p[i] += 1;
        }
    }
    return p;
};


void main() {
    uint globalId = gl_GlobalInvocationID.x;
    // this particle
    vec4 me = attributes[ globalId ];
    // this particle's parameters
    vec2 pos = vec2(me[0], me[1]);
    float charge = me[2];
    float mass = me[3];
    vec2 v = velocities[ globalId ];

    // iterate over all particles to evaluate net force
    uint i;
    vec2 force = vec2(0, 0);
    for (i = uint(0); i < N; i++){
            if (i == globalId){
            continue;
        }
        vec4 other = attributes[i]; // x, y, charge, mass
        vec2 dS = pos - vec2(other[0], other[1]);
        dS = shortest(dS);
        vec2 direction = normalize(dS);
        float distance = max(0.01, length(dS));
        force += sign(charge * other[2]) * direction / pow(distance,2);
    }
    // update velocity and position
    v = (1. - damping) * (v + dt * force / mass);
    pos = bound(pos + dt * v);
    // write back out to the buffers
    attributes[globalId] = vec4(pos, charge, mass);
    velocities[globalId] = v;
}
""" % (NUM_PARTICLES, NUM_PARTICLES)


class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 800, 600

    def __init__(self):
        QGLWidget.__init__(self)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        # Window space to M-space transform parameters.
        self.zoom = 1.
        self.offset = [-0.5, 0.]
        # Set a timer to evaluate more iterations when idle.
        self.idleTimer = QtCore.QTimer()
        self.idleTimer.timeout.connect(self.updateGL)
        self.idleTimer.start(0)


    def makeBuffers(self):
        self.count = N = NUM_PARTICLES
        #self.positions = np.array(np.random.random( (N,2)), dtype=np.float32)
        positions = np.array(np.random.random((N, 2)), dtype=np.float32)
        velocities = np.zeros((N, 2))
        masses = np.ones(N)
        #masses[:] = 50
        masses[-N/2-1:-1] = 1
        charges = -np.ones(N)
        #charges[-N/2-1:-1] = -1
        self.attributes = np.zeros((N, 4), dtype=np.float32)
        self.attributes[:,0:2] = positions
        self.attributes[:,2] = charges
        self.attributes[:,3] = masses
        self.velocities = velocities


    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(1., 1., 1., 0)
        # Make initial data array.
        # compile the vertex shader
        vs = compile_shader(VERTEX, gl.GL_VERTEX_SHADER)
        # compile the fragment shader
        fs = compile_shader(FRAGMENT, gl.GL_FRAGMENT_SHADER)
        # Link the programs.
        self.render_program = link_shaders(vs, fs)
        # Compile the compute shader
        cs = compile_shader(COMPUTE, gl.GL_COMPUTE_SHADER)
        # Create the compute shader buffers.
        self.makeBuffers()
        #self.vbo = glvbo.VBO(self.attributes)
        self.vbo =  gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.attributes.nbytes,
                     self.attributes, gl.GL_DYNAMIC_COPY)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self.ssbo = gl.glGenBuffers(1)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self.ssbo)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, self.velocities.nbytes,
                     self.velocities, gl.GL_DYNAMIC_COPY)
        self.compute_program = link_shaders(cs)  
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_POINT_SMOOTH)  


    def paintGL(self):
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # Run the compute phase
        gl.glUseProgram(self.compute_program)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.vbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self.ssbo)
        loc = gl.glGetUniformLocation(self.compute_program, 'N')
        gl.glUniform1ui(loc, self.count)
        loc = gl.glGetUniformLocation(self.compute_program, 'dt')
        gl.glUniform1f(loc, DT)
        loc = gl.glGetUniformLocation(self.compute_program, 'damping')
        gl.glUniform1f(loc, DAMPING)
        gl.glDispatchCompute(self.count, 1, 1)
        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
        # Read back the modified data.
        #gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,10, self.ssbo)
        #gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0,
        #                      self.velocities.nbytes, self.velocities)

        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        # bind the VBO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        # tell OpenGL that the VBO contains an array of vertices
        # these vertices contain 4 single precision coordinates
        gl.glVertexPointer(4, gl.GL_FLOAT, 0, None)
        # Use our pipeline.
        gl.glUseProgram(self.render_program)
        # draw "count" points from the VBO
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)


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
        gl.glOrtho(-1, 1, 1, -1, -1, 1)


if __name__ == '__main__':
    # import numpy for generating random data points
    import sys
    import numpy as np

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