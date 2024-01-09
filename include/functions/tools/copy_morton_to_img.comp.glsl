#version 460
#extension GL_GOOGLE_include_directive : require
#include "../common_functions.h"

layout(binding = 1, std430) uniform Custom {
    float_ptr in_tensor; // buffer
    float_ptr out_tensor; // img
    int resolution; // power of 2
    int output_dim;
};

int unpart1by1(int n){
    n &= 0x55555555; // base10: 1431655765, binary: 1010101010101010101010101010101,  len: 31
    n = (n ^ (n >> 1)) & 0x33333333; // base10: 858993459,  binary: 110011001100110011001100110011,   len: 30
    n = (n ^ (n >> 2)) & 0x0f0f0f0f; // base10: 252645135,  binary: 1111000011110000111100001111,     len: 28
    n = (n ^ (n >> 4)) & 0x00ff00ff; // base10: 16711935,   binary: 111111110000000011111111,         len: 24
    n = (n ^ (n >> 8)) & 0x0000ffff; // base10: 65535,      binary: 1111111111111111,                 len: 16
    return n;
}

void main(){
    uvec3 thread_id;
    if (!start_function(thread_id))
    return;

    int index = int(thread_id.x);

    int px = unpart1by1(index);
    int py = unpart1by1(index >> 1);

    int offset_src = index * output_dim;
    int offset_dst = (py * resolution + px)*output_dim;

    [[unroll]]
    for (int i=0; i<output_dim; i++)
        out_tensor.data[offset_dst + i] = in_tensor.data[offset_src + i];
}