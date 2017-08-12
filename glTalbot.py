# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
from PyQt4.Qt import *
# PyOpenGL imports
from OpenGL.GL import *
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import math
import os
import ctypes

def compile_shader(shader_type, source_file):
    shader = glCreateShader(shader_type)
    if os.path.isfile(source_file):
        with open(source_file) as f:
            glShaderSource(shader, f.read())
    else:
        glShaderSource(shader, source_file)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != 1:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def link_shaders(*args):
    program = glCreateProgram()
    for shader in args:
        glAttachShader(program, shader)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != 1:
        raise RuntimeError(glGetProgramInfoLog(program))
    return program


class Uniform(object):
    def __init__(self, name, program=None):
        self.name = name
        self.size, self.type, self.index, self.loc = 4*[None]
        if program is None:
            program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
        if program == 0:
            raise Exception("No program specified, and no current program.")
        i = -1
        found = False
        while True:
            i += 1
            try:
                uname, usize, utype = gl.glGetActiveUniform(program, i)
            except:
                break
            uname = uname.decode().split('[')[0]
            if uname == name:
                self.size, self.type = usize, utype
                self.index = i
                self.loc = gl.glGetUniformLocation(program, name)
                found = True
                break
        if not found:
            print("Uniform %s not found." % name)
        else:
            print(name, self.index, self.loc, self.size, self.type)


class GLTalbotWidget(QGLWidget):
    # default window size
    width, height = 600, 600

    class Params(object):
        pass

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc."""
        # background color
        gl.glClearColor(0, 0, 0, 0)
        # compile and link shaders
        vs = compile_shader(GL_VERTEX_SHADER, "glTalbot.vertex.glsl")
        fs = compile_shader(GL_FRAGMENT_SHADER, "glTalbot.fragment.glsl")
        self.shaders_program = link_shaders(vs, fs)
        # create a Vertex Buffer Object with the specified data
        pts = [-1.0, -1.0,
                1.0, -1.0,
               -1.0,  1.0,
                1.0, -1.0,
                1.0,  1.0,
               -1.0,  1.0]
        self.data = (ctypes.c_float * len(pts))(*pts)
        self.vbo = glvbo.VBO(self.data)
        # Create uniforms
        gl.glUseProgram(self.shaders_program)
        self.uniforms = self.__class__.Params()
        self.uniforms.time = Uniform('time')
        self.uniforms.wavelength = Uniform('wavelength')
        self.uniforms.aperture = Uniform('aperture')
        gl.glUniform1f(self.uniforms.wavelength.loc, 5e-2)
        self.set_aperture(0.5, 2)
        # Start redraw timer.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

    def set_aperture(self, width, slits=1):
        gl.glUseProgram(self.shaders_program)
        ap_pts = self.uniforms.aperture.size
        apx = [-width/2 + x*width/(ap_pts-1) for x in range(ap_pts)]
        apy = ap_pts*[1]
        chunk = ap_pts/(2.*slits-1)
        for i in range(1, slits):
            j = 2*i-1
            apy[int(j*chunk):int((j+1)*chunk)] = int(chunk)*[0]
        ap = (ap_pts*(2*ctypes.c_float))(*zip(apx, apy))
        self.aperture = ap
        gl.glUniform2fv(self.uniforms.aperture.loc, ap_pts, ap)

    def set_wavelength(self, wl):
        gl.glUseProgram(self.shaders_program)
        gl.glUniform1f(self.uniforms.wavelength.loc, wl)
        self.update()

    def paintGL(self):
        """Paint the scene."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.vbo.bind()
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        gl.glUseProgram(self.shaders_program)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self.data))

        resLocation = gl.glGetUniformLocation(self.shaders_program, 'resolution')
        gl.glUniform2f(resLocation, self.width, self.height)       

        t = ctypes.c_float()
        gl.glGetUniformfv(self.shaders_program, self.uniforms.time.loc, t)
        gl.glUniform1f(self.uniforms.time.loc, t.value + 3.141/12)

        gl.glUseProgram(0)
        gl.glLineWidth(6)
        gl.glColor3f(0.5, 1.0, 0.5)
        dx = self.aperture[1][0] - self.aperture[0][0]
        for (pt1, pt2) in zip(self.aperture[:-1], self.aperture[1:]):
            if pt1[1] == 1 or pt2[1] == 1:
                continue
            if abs(pt1[0] - pt2[0]) > 2*dx:
                continue
            gl.glBegin(gl.GL_LINES)
            gl.glVertex2f(-1, pt2[0])
            gl.glVertex2f(-1, pt1[0])
            gl.glEnd()

    def resizeGL(self, width, height):
        """Update the window and viewport sizes."""
        self.width, self.height = width, height
        gl.glViewport(0, 0, width, height)
        self.paintGL()
        print("Viewport: %d x %d" % (width, height))


class GLTalbotWindow(QtGui.QWidget):
    def __init__(self):
        super(GLTalbotWindow, self).__init__()
        self.slits = 2
        self.apWidth = 0.5

        glformat = QtOpenGL.QGLFormat()
        glformat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        self.wGl = GLTalbotWidget(glformat)
        wlSlider = QtGui.QSlider(Qt.Horizontal)
        wlSlider.valueChanged.connect(self.set_wavelength)
        self.wlLabel = QtGui.QLabel()

        self.slitLabel = QtGui.QLabel()
        self.apLabel = QtGui.QLabel()
        slitSlider = QtGui.QSlider(Qt.Horizontal)
        slitSlider.setMinimum(1)
        slitSlider.setMaximum(32)
        slitSlider.valueChanged.connect(self.set_slits)
        apSlider = QtGui.QSlider(Qt.Horizontal)
        apSlider.valueChanged.connect(self.set_apwidth)

        # Wrap a vbox in a widget to set a fixed width.
        wControls = QtGui.QWidget()
        wControls.setLayout(QtGui.QVBoxLayout())
        wControls.setMaximumWidth(120)
        wControls.layout().addWidget(self.wlLabel)
        wControls.layout().addWidget(wlSlider)
        wControls.layout().addStretch(1)
        wControls.layout().addWidget(self.slitLabel)
        wControls.layout().addWidget(slitSlider)
        wControls.layout().addStretch(1)
        wControls.layout().addWidget(self.apLabel)
        wControls.layout().addWidget(apSlider)
        wControls.layout().addStretch(1)

        hbox = QtGui.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.wGl)
        hbox.addWidget(wControls)
        hbox.addSpacing(6)

        self.setLayout(hbox)
        self.setGeometry(300, 300, 300, 150)
        self.show()
        wlSlider.setValue(50)
        slitSlider.setValue(2)
        apSlider.setValue(49)

    def set_wavelength(self, position):
        wlMin = math.log10(5e-3)
        wlMax = math.log10(5e-1)
        scale = (wlMax - wlMin) / 99
        newWl = math.pow(10, wlMin+scale*position)
        self.wGl.set_wavelength(newWl)
        self.wlLabel.setText("wl: %.4f" % newWl)

    def set_slits(self, position):
        self.slits = position
        self.slitLabel.setText("n: %d" % self.slits)
        self.wGl.set_aperture(self.apWidth, self.slits)

    def set_apwidth(self, position):
        wMin = math.log10(0.01)
        wMax = math.log10(2)
        scale = (wMax - wMin) / 99
        self.apWidth = math.pow(10, wMin+scale*position)
        self.apLabel.setText("w: %.4f" % self.apWidth)
        self.wGl.set_aperture(self.apWidth, self.slits)


def main():
    import sys
    app = QtGui.QApplication(sys.argv)
    # glformat = QtOpenGL.QGLFormat()
    # glformat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
    w = GLTalbotWindow()
    #w = GLTalbotWidget(glformat)
    #s = QtGui.QSlider()
    #s.show()
    #w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
