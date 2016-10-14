import ctypes

import numpy
from OpenGL.GL import *
from OpenGL.GL import shaders
from PyQt4 import QtCore, QtGui, QtOpenGL


# from PyQt4 import QtGui, QtOpenGL
class MyWidget(QtOpenGL.QGLWidget):
    def initializeGL(self):
        glViewport(0, 0, self.width(), self.height())
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SMOOTH)
        # compile shaders and program
        self.shaderProgram = QtOpenGL.QGLShaderProgram(self)
        self.shaderProgram.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex, "meshVertex.glsl")
        self.shaderProgram.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment, "meshFragment.glsl")
        self.shaderProgram.link()

        vertexPositions = [[0.0, 0.5, 0.0, 1.0],
                           [0.5, -0.366, 0.0, 1.0],
                           [-0.5, -0.366, 0.0, 1.0], ]
        vertexColors = [[1., 0., 0., 1.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 1.], ]


        self.vbo = QtOpenGL.QGLBuffer(GL_ARRAY_BUFFER)



        self.shaderProgram.enableAttributeArray("position")
        self.shaderProgram.enableAttributeArray("color")
        self.shaderProgram.setAttributeArray("position", vertexPositions)
        self.shaderProgram.setAttributeArray("color", vertexColors)

        cs = glCreateShader(GL_COMPUTE_SHADER)
        with open("meshCompute.glsl") as f:
            glShaderSource(cs, f.read())
        glCompileShader(cs)
        if glGetShaderiv(cs, GL_COMPILE_STATUS) != 1:
            raise RuntimeError(glGetShaderInfoLog(cs))

        self.ssbo = glGenBuffers(1)
        self.vertices = numpy.array(vertexPositions, dtype=numpy.float32)
        self.data = numpy.array([1,2,3,4]*4, dtype=numpy.float32)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.vertices.nbytes, self.vertices, GL_DYNAMIC_COPY)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.data.nbytes, self.data, GL_DYNAMIC_COPY)

        self.computeProgram = glCreateProgram()
        glAttachShader(self.computeProgram, cs)
        glLinkProgram(self.computeProgram)
        print glGetProgramiv(self.computeProgram, GL_LINK_STATUS)
        if glGetProgramiv(self.computeProgram, GL_LINK_STATUS) != 1:
            raise RuntimeError(glGetProgramInfoLog(self.computeProgram))


    def paintGL(self):
        glClearColor(0., 0., 0., 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        self.shaderProgram.bind()
        glDrawArrays(GL_POINTS, 0, 3)
        self.shaderProgram.release()

        print(self.data)
        print(self.vertices)
        glUseProgram(self.computeProgram)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ssbo)
        glDispatchCompute(len(self.data), 1, 1)

        #glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           self.vertices.nbytes, self.vertices)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1, self.ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                           self.data.nbytes, self.data)

        print(self.data)
        print(self.vertices)


def main():
    import sys

    app = QtGui.QApplication(sys.argv)

    glformat = QtOpenGL.QGLFormat()
    glformat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
    #glformat.setVersion(4, 0)
    print (glformat.majorVersion(), glformat.minorVersion())
    w = MyWidget(glformat)
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()