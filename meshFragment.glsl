#version 430

//smooth in vec4 theColor;
in vec4 col;
out vec4 outputColor;

void main()
{
    outputColor = col;
    //outputColor = vec4(1.0,1.0,1.0,0.5);
}