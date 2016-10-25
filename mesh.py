NUM_NODES = 256
NUM_DIMS = 2
import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL import shaders
from PyQt4 import QtCore, QtGui, QtOpenGL

# Indexes into a half matrix.
def ijtok(i, j):
    return i - j-1 + j*NUM_NODES - j*(j+1)/2

def ktoij(k):
    N = NUM_NODES
    j = int((2*N - 1 - np.sqrt(1 + 4*N**2 - 4*N - 8* k))/2)
    i = j+1 + k - j*N + j*(j+1)/2
    return i, j

try:
    assert [ijtok(*ktoij(x)) for x in range((NUM_NODES**2-NUM_NODES)/2)] == range((NUM_NODES**2-NUM_NODES)/2)
except AssertionError:
    raise Exception("Incorrect output from ijtok or ktoij.")


class MeshWidget(QtOpenGL.QGLWidget):
    def initializeGL(self):
        glViewport(0, 0, self.width(), self.height())
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SMOOTH)
        # compile shaders and program
        self.shaderProgram = QtOpenGL.QGLShaderProgram(self)
        self.shaderProgram.addShaderFromSourceFile(QtOpenGL.QGLShader.Vertex, "meshVertex.glsl")
        self.shaderProgram.addShaderFromSourceFile(QtOpenGL.QGLShader.Fragment, "meshFragment.glsl")
        self.shaderProgram.link()

        self.pr_matrix = QtGui.QMatrix4x4(1., 0., 0., -.5 ,
                                          0., 1., 0., -.5,
                                          0., 0., 1.,  0.,
                                          0., 0., 0., 1.)

        # random points and edges
        self.positions = np.random.random((NUM_NODES, 2))
        # Adjacency matrix stores L0.
        self.adjacency = np.random.random((NUM_NODES ** 2 - NUM_NODES) / 2)
        p_edge = 1. -  (1. / NUM_NODES)
        self.adjacency[self.adjacency >= p_edge] = 1.
        self.adjacency[self.adjacency < p_edge] = 0.


        self.edge_indices = np.array([ktoij(k) for k in np.where(self.adjacency>0)[0]]).ravel()
        self.forces = np.zeros(((NUM_NODES ** 2 - NUM_NODES) / 2, 2))

        self.shaderProgram.setAttributeArray("position", self.positions)


    def keyPressEvent(self, event):
        key = event.key()
        handled = True
        if key in (QtCore.Qt.Key_Space,):
            self.calc_force()
        else:
            handled = False

        if handled:
            self.updateGL()
        else:
            event.ignore()

    def calc_force(self):
        for i in xrange(len(self.edge_indices)/2):
            index_b = self.edge_indices[2*i+1]
            index_a = self.edge_indices[2*i]
            delta = self.positions[index_b] - self.positions[index_a]
            l = np.sqrt(delta[0]**2 + delta[1]**2)
            direction = delta / l
            adj_index = ijtok(index_a, index_b)
            self.forces[adj_index] = (l - self.adjacency[adj_index]) * direction
        resultants = np.zeros_like(self.positions)
        for p in xrange(NUM_NODES):
            #print p, ijtok(p, p) + 1, ijtok(p + 1, p + 1)
            if p < NUM_NODES:
                resultants[p] -= self.forces[ijtok(p, p)+1:ijtok(p+1, p+1)+1].sum(axis=0)
            if p > 0:
                indices = [p-1]
                for i in xrange(p-1):
                    indices.append(indices[-1] + NUM_NODES - i - 2)
                resultants[p] += self.forces[indices].sum(axis=0)
            self.positions[p] += 0.01 * resultants[p]
        self.shaderProgram.setAttributeArray("position", self.positions)
        self.shaderProgram.setUniformValueArray("forces", self.forces)

    def paintGL(self):
        glClearColor(0., 0., 0., 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.shaderProgram.bind()
        self.shaderProgram.enableAttributeArray("position")
        self.shaderProgram.setUniformValue("pr_matrix", self.pr_matrix)
        glDrawArrays(GL_POINTS, 0, NUM_NODES)
        glDrawElements(GL_LINES, len(self.edge_indices), GL_UNSIGNED_INT, self.edge_indices)
        self.shaderProgram.release()
        self.calc_force()
        QtCore.QTimer.singleShot(1, self.updateGL)

    def resizeGL(self, width, height):
        glViewport(0, 0, self.width(), self.height())


def main():
    import sys

    app = QtGui.QApplication(sys.argv)

    glformat = QtOpenGL.QGLFormat()
    glformat.setProfile(QtOpenGL.QGLFormat.CoreProfile)
    #glformat.setVersion(4, 0)
    w = MeshWidget(glformat)
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()