#version 330
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout( local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout (std430, binding=0) buffer Positions {
    vec2 positions[];
};

void main() {
    uint globalId = gl_GlobalInvocationID.x;
    vec2 position = positions[globalId];
    positions[0][0] = 0;
    positions[0][1] = 0;
    memoryBarrierBuffer();
}