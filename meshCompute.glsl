#version 330
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout (std430, binding=0) buffer Positions {
    vec4 positions[];
};
layout (std430, binding=1) buffer Attributes {
    float attributes[];
};

layout( local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

void main() {
    uint globalId = gl_GlobalInvocationID.x;
    //attributes[globalId] = attributes[1];
    //attributes[globalId] = 0.;
    positions[0] = vec4(-1., -1, 0., 1.);
}
