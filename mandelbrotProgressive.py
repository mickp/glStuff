# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import ctypes
MAX_WORKGROUPS = 65535
ITER_PER_CYCLE=128

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
attribute vec2 position;
attribute ivec2 status;
out int fs_count;
out int fs_converged;

void main(){
   gl_Position = vec4(position[0], position[1], 0.5, 1.0);
   fs_count = status[0];
   fs_converged = status[1];
}
"""

# Fragment shader
FRAGMENT = """
#version 330
flat in int fs_count;
flat in int fs_converged;

void main(){
    vec3 outerColor1 = vec3(1., 0., 1.);
    vec3 outerColor2 = vec3(0., 1., 1.);
    vec3 color;

    if (fs_converged == 1)
    {
        color = mix(outerColor1, outerColor2, fract(float(fs_count)*0.05));
    } else {
        color = vec3(0.1, 0.1, 0.1);
    }
    gl_FragColor = vec4 (clamp(color, 0.0, 1.0), 1.0);
}
"""

# Compute shader
COMPUTE = """
#version 330
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout( std430, binding=0 ) buffer Values {
    vec4 values[ ]; 
};

layout( std430, binding=1 ) buffer Status {
    ivec2 statuses[ ];
};
 
layout( local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

void main() {
    uint globalId = gl_GlobalInvocationID.x;
    vec4 v = values[ globalId ];
    ivec2 s = statuses[ globalId ];

    float real = v[0];
    float imag = v[1];
    float cReal = v[2];
    float cImag = v[3];

    float r2 = 0.0;
    int iter;

    if (s[1] != 0) return;

    for (iter = 0; iter < 128 && r2 < 4.0; ++iter)
    {
        float tempreal = real;
        real = (tempreal * tempreal) - (imag * imag) + cReal;
        imag = 2.0 * tempreal * imag + cImag;
        r2 = real * real;
    }
    
    if (r2 >= 4.0)
    {
        s[1] = 1;
    }

    s[0] += iter;
    v[0] = real;
    v[1] = imag;

    values[ globalId ] = v;
    statuses[globalId] = s;
}

"""


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
        # Track number of iterations evaluated.
        self.iterations = 0
    

    def mousePressEvent(self, event):
        # Save the last click position.
        self.lastClickPos = event.pos()


    def mouseReleaseEvent(self, event):
        # Current zoom, offset and aspect ratio.
        zoom = self.zoom
        offset = self.offset
        aspect = self.aspect
        if self.lastClickPos == event.pos():
            # Offset and optional zoom out.
            if event.button() == 2:
                # Zoom out on right mouse button.
                self.zoom /= 2
            # widget-space co-ordinates
            wx = event.posF().x()
            wy = event.posF().y()
            # M-space co-ordinates
            mx = (2 * wx / self.width - 1) * (aspect/zoom) + offset[0]
            my = -(2 * wy / self.height - 1) * (1.0/zoom) + offset[1]
            self.offset = [mx, my]
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
            mx = (2 * wxAvg / self.width - 1) * (aspect/zoom) + offset[0]
            my = -(2 * wyAvg / self.height - 1) * (1.0/zoom) + offset[1]
            self.offset = [mx, my]
            self.zoom = newZoom
        # Remake the buffers with new transform.
        self.makeBuffers()


    def makeBuffers(self):
        w = self.width
        h = self.height
        self.aspect = float(w)/h
        # Make a 2D meshgrid.
        wpts = np.linspace(-1, 1, w)
        hpts = np.linspace(-1, 1, h)
        xx,yy = np.meshgrid(hpts, wpts, dtype=np.float32)
        # Making a dstack seems to produce a noncontiguous array.
        vNC = np.dstack((yy.flat, xx.flat)).reshape((w*h, 2))
        # Need to make a contiguous array to pass to the shaders.
        v = np.zeros((w*h, 2), dtype=np.float32)
        v[:,0] = vNC[:,0]
        v[:,1] = vNC[:,1]
        self.vertexBuffer = v
        self.vbo = glvbo.VBO(self.vertexBuffer)
        self.count = v.shape[0]
        # Now make the compute buffers.
        self.valueBuffer = np.zeros((w*h, 4), dtype=np.float32)
        self.valueBuffer[:,0] = vNC[:,0] * self.aspect / self.zoom + self.offset[0]
        self.valueBuffer[:,1] = vNC[:,1] / self.zoom + self.offset[1]
        self.valueBuffer[:,2] = vNC[:,0] * self.aspect / self.zoom + self.offset[0]
        self.valueBuffer[:,3] = vNC[:,1] / self.zoom + self.offset[1]
        self.statusBuffer = np.zeros((w*h, 2), dtype=np.int32)
        self.iterations = 0


    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(0, 0, 0, 0)
        # Make initial data array.
        self.makeBuffers()
        # create a Vertex Buffer Object with the specified data
        self.vbo = glvbo.VBO(self.vertexBuffer)
        # compile the vertex shader
        vs = compile_shader(VERTEX, gl.GL_VERTEX_SHADER)
        # compile the fragment shader
        fs = compile_shader(FRAGMENT, gl.GL_FRAGMENT_SHADER)
        # Link the programs.
        self.render_program = link_shaders(vs, fs)
        # Bind the program so we can set initial parameters.
        gl.glUseProgram(self.render_program)
        # View parameters.
        gl.glPointSize(1.0)
        # Compile the compute shader
        cs = compile_shader(COMPUTE, gl.GL_COMPUTE_SHADER)
        # Create the compute shader buffers.
        self.ssbos = dict(map(None, ('data', 'status'), gl.glGenBuffers(2)))
        self.compute_program = link_shaders(cs)      


    def paintGL(self):
        # clear the buffer
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Run the compute phase
        i = 0
        while (i * MAX_WORKGROUPS < self.count):
            upper = min(self.count, (i+1) * MAX_WORKGROUPS)
            data = self.valueBuffer[i*MAX_WORKGROUPS:upper]
            status = self.statusBuffer[i*MAX_WORKGROUPS:upper]
            
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.ssbos['data'])
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, 
                            data.nbytes,
                            data, gl.GL_DYNAMIC_COPY)

            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self.ssbos['status'])
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, 
                            status.nbytes,
                            status, gl.GL_DYNAMIC_COPY)
            

            gl.glUseProgram(self.compute_program)
            gl.glDispatchCompute(min(MAX_WORKGROUPS, self.count - i * MAX_WORKGROUPS), 1, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)

            # Read back the modified data.
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.ssbos['data'])
            gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 
                            0,
                            data.nbytes,
                            self.valueBuffer[i*MAX_WORKGROUPS])
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, self.ssbos['status'])
            gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 
                            0,
                            status.nbytes,
                            self.statusBuffer[i*MAX_WORKGROUPS])
            i += 1
        self.iterations += ITER_PER_CYCLE

        # bind the VBO
        self.vbo.bind()
        # tell OpenGL that the VBO contains an array of vertices
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        # these vertices contain 2 single precision coordinates
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, self.vbo)
        
        # Bind statuses
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.ssbos['status'])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.statusBuffer.nbytes,
                        self.statusBuffer, gl.GL_DYNAMIC_COPY)
        gl.glVertexAttribIPointer(1, 2, gl.GL_INT, 0, None)
        gl.glEnableVertexAttribArray(1);

        # Use our pipeline.
        gl.glUseProgram(self.render_program)
        # draw "count" points from the VBO
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        # Update the window title.
        self.parent().setWindowTitle("Iterations: %d; zoom; %f" % (self.iterations, self.zoom))

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # update the data points
        self.makeBuffers()
        if self.vbo:
            self.vbo.set_array(self.vertexBuffer)
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