#version 330

in vec4 position;
uniform mat4 gl_ModelViewMatrix;
//uniform uint NUM_NODES;
out int ID;

void main()
{
    gl_Position = gl_ModelViewMatrix * position;
    gl_PointSize = 4;
    //theColor = vec4(forces, 0. ,0.5);
    ID = gl_VertexID;
}