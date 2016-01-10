# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo


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
attribute vec2 position;
out vec2 pos;

void main(){
   pos = position;
   gl_Position = vec4(position[0], position[1], 0.5, 1.0);
}
"""


# Fragment shader
FS = """
#version 330
// Output variable of the fragment shader, which is a 4D vector containing the
// RGBA components of the pixel color.
varying vec2 pos;
uniform vec2 center;
uniform float zoom;
uniform float aspect;

// Main fragment shader function.
void main()
{
    vec2 position = pos;
    int maxIterations = 128;
    vec3 outerColor1 = vec3(1., 0., 1.);
    vec3 outerColor2 = vec3(0., 1., 1.);
    // vec2 center = vec2(0., 0.);
    // float zoom = 2;

    float real = position[0] * (aspect/zoom) + center.x;
    float imag = position[1] * (1.0/zoom) + center.y;
    float cReal = real;
    float cImag = imag;

    float r2 = 0.0;
    int iter;

    for (iter = 0; iter < maxIterations && r2 < 4.0; ++iter)
    {
        float tempreal = real;
        real = (tempreal * tempreal) - (imag * imag) + cReal;
        imag = 2.0 * tempreal * imag + cImag;
        r2 = real * real;
    }
    // Base the color on the number of iterations.
    vec3 color;
    if (r2 < 4.0)
        color = vec3(0.1, 0.1, 0.1);
    else
        color = mix(outerColor1, outerColor2, fract(float(iter)*0.05));
    gl_FragColor = vec4 (clamp(color, 0.0, 1.0), 1.0);
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

    def makeData(self):
        w = self.width
        h = self.height
        wpts = np.linspace(-1, 1, w)
        hpts = np.linspace(-1, 1, h)
        xx,yy = np.meshgrid(hpts, wpts, dtype=np.float32)
        # Making a dstack seems to produce a noncontiguous array.
        dataNC = np.dstack((yy.flat, xx.flat)).reshape((w*h, 2))
        # Need to make a contiguous array to pass to the shaders.
        data = np.zeros((w*h, 2), dtype=np.float32)
        data[:,0] = dataNC[:,0]
        data[:,1] = dataNC[:,1]
        self.data = data
        self.count = data.shape[0]


    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # Make initial data array.
        self.makeData()
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
        # Bind the program so we can set initial parameters.
        gl.glUseProgram(self.shaders_program)
        # View parameters.
        self.uZoom = Uniform(gl.glUniform1f, 
                             self.shaders_program,
                             'zoom',
                              1/2.5)
        self.uCenter = Uniform(gl.glUniform2f, 
                               self.shaders_program,
                               'center',
                               (0, 0))
        self.uAspect = Uniform(gl.glUniform1f, 
                               self.shaders_program,
                               'aspect', 
                               float(self.width) / self.height)


    def mousePressEvent(self, event):
        self.lastClickPos = event.pos()
        

    def mouseReleaseEvent(self, event):
        # Current zoom and offset
        zoom = self.uZoom.value
        offset = self.uCenter.value
        aspect = float(self.width) / float(self.height)
        if self.lastClickPos == event.pos():
            # Offset and optional zoom out.
            if event.button() == 2:
                # Zoom out on right mouse button.
                self.uZoom.update(self.uZoom.value / 2)
            # widget-space co-ordinates
            wx = event.posF().x()
            wy = event.posF().y()
            # GL-space co-ordinates
            gx = (2 * wx / self.width - 1) * (aspect/zoom) + offset[0]
            gy = -(2 * wy / self.height - 1) * (1.0/zoom) + offset[1]
            self.uCenter.update((gx, gy))
        elif event.button() == 1:
            # Zoom and offset, only on left button.
            wx0 = self.lastClickPos.x()
            wy0 = self.lastClickPos.y()
            wx1 = event.posF().x()
            wy1 = event.posF().y()
            wxAvg = 0.5 * (wx0 + wx1)
            wyAvg = 0.5 * (wy0 + wy1)
            fraction = max(abs(wy1 - wy0) / self.height,
                           aspect * abs(wx1 - wx0) / self.width)
            newZoom = zoom / fraction
            gx = (2 * wxAvg / self.width - 1) * (aspect/zoom) + offset[0]
            gy = -(2 * wyAvg / self.height - 1) * (1.0/zoom) + offset[1]
            self.uCenter.update((gx, gy))
            self.uZoom.update(newZoom)

        self.updateGL()


    def paintGL(self):
        """Paint the scene.
        """
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # set yellow color for subsequent drawing rendering calls
        gl.glColor(1,1,0)
        # bind the VBO
        self.vbo.bind()
        # tell OpenGL that the VBO contains an array of vertices
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        # these vertices contain 2 single precision coordinates
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, self.vbo)
        # Use our pipeline.
        gl.glUseProgram(self.shaders_program)
        # draw "count" points from the VBO
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)


    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        self.uAspect.update(float(width) / float(height))
        # update the data points
        self.makeData()
        if self.vbo:
            self.vbo.set_array(self.data)
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