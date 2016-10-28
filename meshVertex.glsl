#version 430

in vec4 position;
uniform mat4 gl_ModelViewMatrix;
out vec4 col;

void main()
{
    gl_Position = gl_ModelViewMatrix * position;
    gl_PointSize = 12;
    col = vec4(0.5, 1., 0.5, 0.5);
}