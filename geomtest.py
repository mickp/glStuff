"""The geometry shader should populate the feedback buffer with
vertex indices. However, it only does this if both EmitVertex calls
are commented out."""

NUM_NODES = 6
import numpy as np
from OpenGL.GL import *
from PyQt4 import QtCore, QtGui, QtOpenGL

VS = """#version 440

in vec4 position;
out VSOUT
{
  vec4 gl_Position;
  int index;
} vsout;
uniform mat4 gl_ModelViewMatrix;


void main()
{
    gl_Position = gl_ModelViewMatrix * position;
    vsout.index = gl_VertexID;
    vsout.gl_Position = gl_Position;
}
"""

GS = """#version 440
#extension GL_ARB_shader_storage_buffer_object : enable

layout (lines) in;
layout (line_strip) out;
in VSOUT{
  vec4 gl_Position;
  int index;
} vdata[];

layout (std430, binding=0) buffer FeedbackBuffer{
    vec2 fb[];
};

void main()
{
  int i = vdata[0].index;
  int j = vdata[1].index;

  fb[gl_PrimitiveIDIn ][0] = vdata[0].index;
  fb[gl_PrimitiveIDIn ][1] = vdata[1].index;

  gl_Position = gl_in[0].gl_Position;
  //EmitVertex();
  gl_Position = gl_in[1].gl_Position;
  //EmitVertex();
}
"""

FS = """#version 440
out vec4 outputColor;

void main()
{
    outputColor = vec4(.5,.5,.5,.5);
}
"""

class GeomTestWidget(QtOpenGL.QGLWidget):
    def initializeGL(self):
        glViewport(0, 0, self.width(), self.height())
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SMOOTH)
        self.shaderProgram = QtOpenGL.QGLShaderProgram(self)
        self.shaderProgram.addShaderFromSourceCode(QtOpenGL.QGLShader.Vertex, VS)
        self.shaderProgram.addShaderFromSourceCode(QtOpenGL.QGLShader.Geometry, GS)
        self.shaderProgram.addShaderFromSourceCode(QtOpenGL.QGLShader.Fragment, FS)
        self.shaderProgram.link()

        self.pr_matrix = [1., 0., 0., -.0 ,
                          0., 1., 0., -.0,
                          0., 0., 1.,  0.,
                          -.5, -.5, 0., 1.]

        # random points and edges
        self.positions = np.random.random((NUM_NODES, 2))
        self.fbBuffer = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.fbBuffer)
        self.fb = np.zeros((NUM_NODES,2), dtype=np.float32)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.fb.nbytes, self.fb, GL_DYNAMIC_COPY)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.fbBuffer)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def keyPressEvent(self, event):
        key = event.key()
        handled = True
        if key in (QtCore.Qt.Key_Space,):
            self.getfb()
        else:
            handled = False

        if handled:
            self.updateGL()
        else:
            event.ignore()

    def getfb(self, init=False):
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.fbBuffer)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.fb.nbytes, self.fb)
        print self.fb

    def paintGL(self):
        glClearColor(0., 0., 0., 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(self.pr_matrix)

        self.shaderProgram.bind()
        self.shaderProgram.setAttributeArray("position", self.positions)
        self.shaderProgram.setUniformValue("NUM_NODES", NUM_NODES)
        self.shaderProgram.enableAttributeArray("position")
        # draw nodes
        #glDrawArrays(GL_POINTS, 0, NUM_NODES)
        # draw edges
        edges = [0,1,0,2,2,3,2,4]
        glDrawElements(GL_LINES, len(edges), GL_UNSIGNED_INT, edges)
        self.shaderProgram.release()

    def resizeGL(self, width, height):
        glViewport(0, 0, self.width(), self.height())


def main():
    import sys

    app = QtGui.QApplication(sys.argv)

    glformat = QtOpenGL.QGLFormat()
    glformat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
    #glformat.setVersion(4, 0)
    w = GeomTestWidget(glformat)
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()