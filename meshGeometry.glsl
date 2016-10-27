#version 330
#extension GL_ARB_shader_storage_buffer_object : enable

layout (lines) in;
layout (line_strip) out;
uniform int NUM_NODES;
out vec4 col;

layout (std430) buffer StrainsBuffer{
    float strains[];
};

void main()
{
  for(int m = 0; m < gl_in.length(); m++)
  {
     // copy attributes
    gl_Position = gl_in[m].gl_Position;
    float strain = strains[gl_PrimitiveIDIn];
    if (strain >=0) {
        col = vec4(strain, 0.5*(1-strain), 0., 1.);
    } else {
        col = vec4(0., 0.5*(1+strain), -strain, 1.);
    }
    EmitVertex();
  }
}