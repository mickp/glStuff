NUM_NODES = 128
RIGHT = 1.
LEFT = -1.
TOP = 1.
BOTTOM = -1.
NEAR = 1
FAR = -1

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

def compile_shader(shader_type, source_file):
    shader = glCreateShader(shader_type)
    with open(source_file) as f:
        glShaderSource(shader, f.read())
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


class MeshWidget(QtOpenGL.QGLWidget):
    useComputeShader = True;
    perspective = False
    transpose = True
    def initializeGL(self):
        glViewport(0, 0, self.width(), self.height())
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SMOOTH)
        buffers = glGenBuffers(5)
        self.buffers = {'strains':buffers[0],
                        'positions':buffers[1],
                        'lengths':buffers[2],
                        'edges':buffers[3],
                        'forces':buffers[4],}
        # random points and edges
        self.positions = np.random.random((NUM_NODES, 2)).astype(np.float32) - 0.5
        # Adjacency matrix stores L0.
        self.adjacency = np.random.random((NUM_NODES ** 2 - NUM_NODES) / 2)
        p_edge = 0.#1. -  (1. / NUM_NODES)
        self.adjacency[self.adjacency >= p_edge] = 1.
        self.adjacency[self.adjacency < p_edge] = 0.
        # Interaction forces
        self.forces = np.zeros(((NUM_NODES ** 2 - NUM_NODES) / 2, 2), dtype=np.float32)
        # Indices to two points per edge.
        self.edge_indices = np.array([ktoij(k) for k in np.where(self.adjacency > 0)[0]],
                                     dtype=np.int).ravel()
        # Strains.
        self.strains = np.zeros(((NUM_NODES ** 2 - NUM_NODES) / 2), dtype=np.float32)
        # Lengths
        self.lengths = np.ones_like(self.strains)
        #self.lengths[range(0, len(self.lengths), 2)] = 0.5
        # Initialize buffers.
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['strains'])
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.strains.nbytes, self.strains, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['lengths'])
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.lengths.nbytes, self.lengths, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['positions'])
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['edges'])
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.edge_indices.nbytes, self.edge_indices, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['forces'])
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.forces.nbytes, self.forces, GL_DYNAMIC_COPY)

        self.shaders = {}
        self.shaders['vertex'] = compile_shader(GL_VERTEX_SHADER, "meshVertex.glsl")
        self.shaders['geometry'] = compile_shader(GL_GEOMETRY_SHADER, "meshGeometry.glsl")
        self.shaders['fragment'] = compile_shader(GL_FRAGMENT_SHADER, "meshFragment.glsl")
        self.shaders['compute'] = compile_shader(GL_COMPUTE_SHADER, "meshCompute.glsl")

        self.computeProgram = link_shaders(self.shaders['compute'])

        self.renderEdges = link_shaders(self.shaders['vertex'],
                                        self.shaders['geometry'],
                                        self.shaders['fragment'])

        self.renderNodes = link_shaders(self.shaders['vertex'],
                                        self.shaders['fragment'])

        #glUseProgram(self.renderNodes)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers['positions'])
        glBufferData(GL_ARRAY_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_DRAW)

    def getProjection(self):
        l = float(LEFT)
        r = float(RIGHT)
        t = float(TOP)
        b = float(BOTTOM)
        n = float(NEAR)
        f = float(FAR)
        if MeshWidget.perspective:
            matrix = [2*n/(r-l),         0,  (r+l)/(r-l),           0,
                              0, 2*n/(t-b),  (t+b)/(t-b),           0,
                              0,         0, -(f+n)/(f-n), -2*f*n/(f-n),
                              0,         0,           -1,           0]
        else:
            matrix = [2/(r-l),          0,        0, -(r+l)/(r-l),
                            0,    2/(t-b),        0, -(t+b)/(t-b),
                            0,          0, -2/(f-n), -(f+n)/(f-n),
                            0,          0,        0,            1,]
        matrix = np.array(matrix, dtype=np.float32)
        matrix.reshape((4,4))
        return matrix



    def keyPressEvent(self, event):
        key = event.key()
        handled = True
        if key in (QtCore.Qt.Key_Space,):
            self.calc_force()
        elif key in (QtCore.Qt.Key_C,):
            MeshWidget.useComputeShader = not MeshWidget.useComputeShader
            print "useComputeShader = ", MeshWidget.useComputeShader
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['positions'])
            if MeshWidget.useComputeShader:
                glBufferData(GL_SHADER_STORAGE_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_COPY)
            else:
                glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.positions.nbytes, self.positions)
        elif key in (QtCore.Qt.Key_P,):
            MeshWidget.perspective = not MeshWidget.perspective
            print "Projection ", self.getProjection()
        elif key in (QtCore.Qt.Key_T,):
            MeshWidget.transpose = not MeshWidget.transpose
            print "Transpose ", MeshWidget.transpose
        else:
            handled = False

        if True:#handled:
            self.updateGL()
        else:
            event.ignore()

    def calc_force(self):
        for i in xrange(len(self.edge_indices) / 2):
            index_b = self.edge_indices[2 * i + 1]
            index_a = self.edge_indices[2 * i]
            delta = self.positions[index_b] - self.positions[index_a]
            l = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
            direction = delta / l
            adj_index = ijtok(index_a, index_b)
            self.strains[i] = l - self.adjacency[adj_index]
            self.forces[adj_index] = self.strains[i] * direction
        resultants = np.zeros_like(self.positions)
        for p in xrange(NUM_NODES):
            if p < NUM_NODES:
                resultants[p] -= self.forces[ijtok(p, p) + 1:ijtok(p + 1, p + 1) + 1].sum(axis=0)
            if p > 0:
                indices = [p - 1]
                for i in xrange(p - 1):
                    indices.append(indices[-1] + NUM_NODES - i - 2)
                resultants[p] += self.forces[indices].sum(axis=0)
            self.positions[p] += 0.01 * resultants[p]
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['strains'])
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.strains.nbytes, self.strains, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.buffers['positions'])
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.positions.nbytes, self.positions, GL_DYNAMIC_DRAW)

    def paintGL(self):
        glClearColor(0., 0., 0., 1.)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        proj = self.getProjection()
        # Render phase.
        glUseProgram(self.renderNodes)
        glUniformMatrix4fv(0, 1, MeshWidget.transpose, proj)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers['positions'])
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glDrawArrays(GL_POINTS, 0, NUM_NODES)

        glUseProgram(self.renderEdges)
        glUniformMatrix4fv(0, 1, MeshWidget.transpose, proj)
        glVertexPointer(2, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers['positions'])
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.buffers['strains'])
        #glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)
        glEnable(GL_RESCALE_NORMAL)
        glDrawElements(GL_LINES, len(self.edge_indices), GL_UNSIGNED_INT, self.edge_indices)


        if MeshWidget.useComputeShader:
            glUseProgram(self.computeProgram)
            # Compute phase.
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.buffers['strains'])
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.buffers['positions'])
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.buffers['lengths'])
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.buffers['edges'])
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.buffers['forces'])
            glDispatchCompute(max(NUM_NODES, len(self.edge_indices)/2), 1, 1)
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)
        else:
            self.calc_force()

        QtCore.QTimer.singleShot(0, self.updateGL)


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