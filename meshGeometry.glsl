#version 330
#extension GL_ARB_shader_storage_buffer_object : enable

layout (lines) in;
layout (line_strip) out;
uniform uint NUM_NODES;
in int ID[2];
out vec4 col;

layout (std430) buffer ForcesBuffer{
    vec2 forces[];
};

void main()
{
  for(int m = 0; m < gl_in.length(); m++)
  {
     // copy attributes
    gl_Position = gl_in[m].gl_Position;
    int i = ID[0];
    int j = ID[1];
    int k = i - j-1 + j*int(NUM_NODES) - j*(j+1)/2;
    vec2 f = clamp(abs(forces[k]), 0, 1);
    col = vec4(f, 0.5, 1.);
    //col = vec4(.4, .6, 0.5, 0.5);
    // done with the vertex
    EmitVertex();
  }
}