#version 430
#extension GL_ARB_shader_storage_buffer_object : enable

layout (lines) in;
layout (line_strip, max_vertices=2) out;
uniform int NUM_NODES;
out vec4 col;

layout (std430, binding=0) coherent buffer Strains{
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
        col = vec4(strain, 1-strain, 0., 1.);
    } else {
        col = vec4(0., 1+strain, -strain, 1.);
    }
    EmitVertex();
  }
}