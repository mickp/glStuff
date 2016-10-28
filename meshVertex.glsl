#version 430

in vec4 position;
uniform layout (location=0) mat4 modelViewMatrix;
out vec4 col;

void main()
{
    gl_Position = modelViewMatrix * position;
    gl_PointSize = 12;
    col = vec4(0.5, 1., 0.5, 0.5);
}