#version 430
#extension GL_ARB_compute_shader : enable
#extension GL_ARB_shader_storage_buffer_object : enable

layout( local_size_x = 1, local_size_y = 1, local_size_z = 1 ) in;

layout (std430, binding=0) coherent buffer Strains {
    float strains[];
};
layout (std430, binding=1) buffer Positions {
    vec2 positions[];
};
layout (std430, binding=2) buffer Lengths {
    float lengths[];
};
layout (std430, binding=3) buffer Edges {
    int edges[];
};
layout (std430, binding=4) buffer Forces {
    vec2 forces[];
};

int ijtok(uint ui, uint uj) {
    int i = int(ui);
    int j = int(uj);
    return i - j-1 + j*positions.length() - j*(j+1)/2;
    }

void main() {
    uint globalId = gl_GlobalInvocationID.x;

    if (globalId < edges.length()/2){
        vec2 delta = positions[edges[2*globalId+1]] - positions[edges[2*globalId]];
        strains[globalId] = length(delta) - lengths[globalId];
        forces[globalId] = strains[globalId] * normalize(delta);
    }

    memoryBarrierBuffer();
    vec2 resultant = vec2(0., 0.);
    if (globalId < positions.length()) {
        int start = ijtok(globalId, globalId) + 1;
        int end = ijtok(globalId+1, globalId+1);
        for (int k = start; k <= end; k++){
            resultant -= forces[k];
        }
        if (globalId > 0) {
            int k = int(globalId) - 1;
            for (int n = 0; n < globalId; n++) {
                resultant += forces[k];
                k += positions.length() - n - 2;
            }
        }
    }
    memoryBarrierBuffer();
    if (globalId < positions.length()) {
        positions[globalId] += 0.001 * resultant;
    }
}
