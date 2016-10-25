#version 330

uniform mat4 pr_matrix;
in vec4 position;
//layout (location=1) in vec4 color;

//smooth out vec4 theColor;

void main()
{
    gl_Position = pr_matrix * position;//*pr_matrix;
    gl_PointSize = 4;
    //theColor = color;
}