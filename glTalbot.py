# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
from OpenGL.GL import *
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
import os
import ctypes

AP_PTS = 32
AP_WIDTH = 1e-3

# Window creation function.
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
            print (name, self.index, self.loc, self.size, self.type)


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
        self.uniforms = __class__.Params()
        self.uniforms.time = Uniform('time')
        self.uniforms.wavelength = Uniform('wavelength')
        self.uniforms.aperture = Uniform('aperture')
        apx = [-AP_WIDTH/2 + x*AP_WIDTH/(AP_PTS-1) for x in range(AP_PTS)]
        apx [AP_PTS//4:3*AP_PTS//4] = [0]*(AP_PTS//2)
        apy = AP_PTS*[1]
        ap = (AP_PTS*(2*ctypes.c_float))(*zip(apx, apy))
        gl.glUniform2fv(self.uniforms.aperture.loc, AP_PTS, ap)
        # Start redraw timer.
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

    def paintGL(self):
        """Paint the scene."""
        #gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.vbo.bind()
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)
        gl.glUseProgram(self.shaders_program)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self.data))

        resLocation = gl.glGetUniformLocation(self.shaders_program, 'resolution')
        gl.glUniform2f(resLocation, self.width, self.height)       

        t = ctypes.c_float()
        gl.glGetUniformfv(self.shaders_program, self.uniforms.time.loc, t)
        gl.glUniform1f(self.uniforms.time.loc, t.value + 0.1)


    def resizeGL(self, width, height):
        """Update the window and viewport sizes."""
        self.width, self.height = width, height
        gl.glViewport(0, 0, width, height)


def main():
    import sys
    app = QtGui.QApplication(sys.argv)
    glformat = QtOpenGL.QGLFormat()
    glformat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
    w = GLTalbotWidget(glformat)
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
