# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import ctypes
import struct
DT = 0.00001
DAMPING = 0.001
NUM_PARTICLES = 512
RM = 0.05
FRAC = 0.1


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
attribute vec4 velocity;
flat out float gs_charge;
flat out float gs_mass;
flat out vec4 gs_velocity;

void main(){
    vec2 position = vec2(attr[0], attr[1]);
    float charge = attr[2];
    float mass = attr[3];
    gl_Position = vec4(2*position-1, 0.5, 1.0);
    gs_charge = charge;
    gs_mass = mass;
    gs_velocity = velocity; 
}
"""

# Geometry shader
GEOMETRY = """
#version 330
#define CIRCLE_SECTIONS 12
#define VERTICES 39 // (12+1) * 3
#define PI 3.1415926

layout (points) in;
flat in float gs_charge[];
flat in float gs_mass[];
flat in vec4 gs_velocity[];
layout (triangle_strip, max_vertices=VERTICES) out;
flat out float fs_charge;
flat out vec4 fs_velocity;
uniform float aspect;

void main(){
    fs_charge = gs_charge[0];
    fs_velocity = gs_velocity[0];
    float r =  0.002 * (log(gs_mass[0]) + 2);
    // Central vertex
    vec4 centre = gl_in[0].gl_Position;

    for (int i=1; i<=CIRCLE_SECTIONS+1; i++) {
        vec4 last_offset;
        // Angle between each side in radians
        float ang = PI * 2.0 / float(CIRCLE_SECTIONS) * i;

        // Offset from center
        vec4 offset = vec4(cos(ang) * r, -sin(ang) * r * aspect, 0.0, 0.0);
        if (i > 1) {
            gl_Position = centre;
            EmitVertex();
            gl_Position = gl_in[0].gl_Position + last_offset;
            EmitVertex();
            gl_Position = gl_in[0].gl_Position + offset;
            EmitVertex();
        }
        last_offset = offset;
    }
    EndPrimitive();

    // Arrow to show force.
    vec4 force = vec4(gs_velocity[0][2], gs_velocity[0][3], 0., 0.);
    //force = normalize(force);
    // perpendicularity:  [-b, a] is perpendicular to [a, b]
    vec4 perp_force = normalize(vec4(-force[1], force[0], 0., 0.));
    gl_Position = centre - r * perp_force;
    EmitVertex();
    gl_Position = centre + 0.005 * log(length(force/gs_mass[0])) * normalize(force);
    EmitVertex();
    gl_Position = centre + r * perp_force;
    EmitVertex();
    EndPrimitive();

}

"""


# Fragment shader
FRAGMENT = """
#version 330
flat in float fs_charge;
flat in vec4 fs_velocity;
const vec2 vecR = normalize(vec2(0. , 1));     // 12 o'clock
const vec2 vecG = normalize(vec2(1.732, -1));  // 4 o'clock
const vec2 vecB = normalize(vec2(-1.732, -1)); // 8 o'clock

void main(){
    if (fs_charge > 0){
        gl_FragColor = vec4(1., 0., 0., 1.);
    } else if (fs_charge < 0) {
        // gl_FragColor = vec4(0., 0., 1., 1.);
        vec2 v = normalize(vec2(fs_velocity[0], fs_velocity[1]));
        gl_FragColor = vec4(dot(v, vecR), dot(v, vecG), dot(v, vecB), 1.);
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
    vec4 velocities[%d];
};

uniform uint N;
uniform float dt;
uniform float damping;
uniform float rm;

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
        if (p[i] >= 1)
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
    vec2 v = vec2(velocities[ globalId ][0], velocities[ globalId ][1]);

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
        float distance = length(dS);
        force += 1000 * direction * (pow(rm/distance, 12) - 2. * pow(rm/distance, 6));
        force += sign(charge * other[2]) * direction / pow(distance,2);
    }
    force = clamp(force, -mass/dt, mass/dt);
    // update velocity and position
    v = (1. - damping) * v + (dt * force / mass);
    pos = bound(pos + dt * v);
    // write back out to the buffers
    attributes[globalId] = vec4(pos, charge, mass);
    velocities[globalId] = vec4(v, force);
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
        positions = np.array(np.random.random((N, 2)), dtype=np.float32)
        velocities = np.zeros((N, 4))
        masses = np.ones(N)
        charges = -np.ones(N)
        if FRAC > 0:
            masses[0:int(N*FRAC)] = 20000
            charges[0:int(N*FRAC)] = 8
        elif FRAC < 0:
            masses[0:int(-FRAC)] = 20000
            charges[0:int(-FRAC)] = 8
        self.attributes = np.zeros((N, 4), dtype=np.float32)
        self.attributes[:,0:2] = positions
        self.attributes[:,2] = charges
        self.attributes[:,3] = masses
        self.velocities = velocities


    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(0.8, 0.8, 0.8, 0)
        # Make initial data array.
        # compile the vertex shader
        vs = compile_shader(VERTEX, gl.GL_VERTEX_SHADER)
        # compile the geometry shader
        gs = compile_shader(GEOMETRY, gl.GL_GEOMETRY_SHADER)
        # compile the fragment shader
        fs = compile_shader(FRAGMENT, gl.GL_FRAGMENT_SHADER)
        # Link the programs.
        self.render_program = link_shaders(vs, gs, fs)
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


    def paintGL(self):
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # Run the compute phase
        try:
            gl.glUseProgram(self.compute_program)
        except:
            import sys
            sys.exit()
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.vbo)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self.ssbo)
        loc = gl.glGetUniformLocation(self.compute_program, 'N')
        gl.glUniform1ui(loc, self.count)
        loc = gl.glGetUniformLocation(self.compute_program, 'dt')
        gl.glUniform1f(loc, DT)
        loc = gl.glGetUniformLocation(self.compute_program, 'damping')
        gl.glUniform1f(loc, DAMPING)
        loc = gl.glGetUniformLocation(self.compute_program, 'rm')
        gl.glUniform1f(loc, RM)
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
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.ssbo)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1);
        # Use our pipeline.
        gl.glUseProgram(self.render_program)
        loc = gl.glGetUniformLocation(self.render_program, 'aspect')
        gl.glUniform1f(loc, float(self.width) / self.height)
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