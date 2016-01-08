# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo


# Window creation function.
def create_window(window_class):
    """Create a Qt window in Python, or interactively in IPython with Qt GUI
    event loop integration:
        # in ~/.ipython/ipython_config.py
        c.TerminalIPythonApp.gui = 'qt'
        c.TerminalIPythonApp.pylab = 'qt'
    See also:
        http://ipython.org/ipython-doc/dev/interactive/qtconsole.html#qt-and-the-qtconsole
    """
    app_created = False
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
        app_created = True
    app.references = set()
    window = window_class()
    app.references.add(window)
    window.show()
    if app_created:
        app.exec_()
    return window

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
VS = """#version 110

uniform float timer;
attribute vec4 position;
varying vec3 texcoord;

mat4 view_frustum(
    float angle_of_view,
    float aspect_ratio,
    float z_near,
    float z_far
) {
    return mat4(
        vec4(1.0/tan(angle_of_view),           0.0, 0.0, 0.0),
        vec4(0.0, aspect_ratio/tan(angle_of_view),  0.0, 0.0),
        vec4(0.0, 0.0,    (z_far+z_near)/(z_far-z_near), 1.0),
        vec4(0.0, 0.0, -2.0*z_far*z_near/(z_far-z_near), 0.0)
    );
}

mat4 scale(float x, float y, float z)
{
    return mat4(
        vec4(x,   0.0, 0.0, 0.0),
        vec4(0.0, y,   0.0, 0.0),
        vec4(0.0, 0.0, z,   0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
}

mat4 translate(float x, float y, float z)
{
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(x,   y,   z,   1.0)
    );
}

mat4 rotate_x(float theta)
{
    return mat4(
        vec4(1.0,       0.0,        0.0,        0.0),
        vec4(0.0,       cos(theta), sin(-theta),0.0),
        vec4(0.0,       sin(theta), cos(theta), 0.0),
        vec4(0.0,       0.0,        0.0,        1.0)
    );
}

mat4 rotate_y(float theta)
{
    return mat4(
        vec4(cos(theta),    0.0,        sin(theta), 0.0),
        vec4(0.0,           1.0,        0.0,        0.0),
        vec4(-sin(theta),   0.0,         cos(theta), 0.0),
        vec4(0.0,           0.0,        0.0,        1.0)
    );
}

void main()
{
    gl_Position = view_frustum(radians(45.0), 1.0, 0.0, 5.0)
        * translate(0.5, 1., -10.0)
        * rotate_y(45.0)
        * scale(1.0, 1.0, 1.0)
        * rotate_x(mod(timer/10.,360.))
        * vec4(position.x, sin(timer+10.*position.x)/4., position.z, position.w);

    //texcoord = position.xy * vec2(0.5) + vec2(0.5);
    texcoord = vec3(position.x, position.y, position.z) * vec3(0.5) + vec3(0.5);
}
"""


"""
    gl_Position = view_frustum(radians(45.0), 4.0/3.0, 0.5, 5.0)
        * translate(0.0, 0.0, 3.0) 
        * rotate_x(timer)
        * scale(4.0/3.0, 1.0, 1.0* cos(timer)) 
        * position;
"""


# Fragment shader
FS = """
#version 330
// Output variable of the fragment shader, which is a 4D vector containing the
// RGBA components of the pixel color.
in vec3 texcoord;
out vec4 out_color;
uniform float timer;

// Main fragment shader function.
void main()
{
    out_color = vec4(texcoord.x, texcoord.y, texcoord.z, 1.);
}
"""

class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 600, 600

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(0, 0, 0, 0)
        # create a Vertex Buffer Object with the specified data
        self.vbo = glvbo.VBO(self.data)
        # compile the vertex shader
        vs = compile_vertex_shader(VS)
        # compile the fragment shader
        fs = compile_fragment_shader(FS)
        # compile the vertex shader
        self.shaders_program = link_shader_program(vs, fs)
        self.currentTime = (gl.glGetUniformLocation(self.shaders_program, 'timer'), 0.0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)
    

    def paintGL(self):
        """Paint the scene."""
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # bind the VBO 
        self.vbo.bind()
        # tell OpenGL that the VBO contains an array of vertices
        # prepare the shader        
        gl.glEnableVertexAttribArray(0)
        # these vertices contain 2 single precision coordinates
        gl.glVertexAttribPointer(0, 4, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glUseProgram(self.shaders_program)
        # draw "count" points from the VBO
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, len(self.data))
        #gl.glDrawArrays(gl.GL_POINTS, 0, len(self.data))
        t = self.currentTime[1] + 0.1
        self.currentTime = (self.currentTime[0], t)
        gl.glUniform1f(self.currentTime[0], t)


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport."""
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)


if __name__ == '__main__':
    # import numpy for generating random data points
    import sys
    import numpy as np

    # null signal
    data = np.zeros((128, 4), dtype=np.float32)
    data[:,0] = np.linspace(-1., 1., len(data))
    data[:,1] = np.sin(8*data[:,0])
    data[0::2,2] = -0.05
    data[1::2,2] = 0.05
    data[:,3] = -.1

    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            # initialize the GL widget
            self.widget = GLPlotWidget()
            self.widget.data = data
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    # show the window
    win = create_window(TestWindow)