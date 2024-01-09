#ifndef COMMON_H
#define COMMON_H

#include "supported.h"

// #define SUPPORTED_FLOAT_ATOM_ADD

#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference2: require
#extension GL_ARB_gpu_shader_int64 : require
#ifdef SUPPORTED_FLOAT_ATOM_ADD
#extension GL_EXT_shader_atomic_float : require
#endif
#extension GL_EXT_control_flow_attributes : require

#include "randoms.h"
#include "scattering_AD.h"

#define FORWARD void forward(map_object, float _input[INPUT_DIM], out float _output[OUTPUT_DIM])
#define BACKWARD void backward(map_object, float _input[INPUT_DIM], float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM])

#define SAVE_SEED(v) uvec4 _rdv_saved_seed_##v = get_seed();
#define SET_SEED(v) set_seed(_rdv_saved_seed_##v);
#define BRANCH_SEED(v) uvec4 _rdv_saved_seed_##v = create_branch_seed();

#define SWITCH_SEED(from_seed, to_seed) \
_rdv_saved_seed_##from_seed = get_seed(); \
set_seed(_rdv_saved_seed_##to_seed);

#define USING_SEED(secondary_seed, method_call) \
set_seed(_rdv_saved_seed_##secondary_seed); \
method_call;

#define USING_SECONDARY_SEED(main_seed, secondary_seed, method_call) \
_rdv_saved_seed_##main_seed = get_seed(); \
set_seed(_rdv_saved_seed_##secondary_seed); \
method_call; \
_rdv_saved_seed_##secondary_seed = get_seed(); \
set_seed(_rdv_saved_seed_##main_seed);



#define GPUPtr uint64_t

#define POINTER(type_name, align) layout(buffer_reference, scalar, buffer_reference_align=align) buffer type_name##_ptr { type_name data[]; };

POINTER(float, 4)
POINTER(int, 4)
POINTER(uint, 4)
POINTER(GPUPtr, 8)
POINTER(vec4, 4)
POINTER(vec3, 4)
POINTER(vec2, 4)
POINTER(mat2, 4)
POINTER(mat3, 4)
POINTER(mat4, 4)
POINTER(mat2x3, 4)
POINTER(mat2x4, 4)
POINTER(mat3x2, 4)
POINTER(mat3x4, 4)
POINTER(mat4x2, 4)
POINTER(mat4x3, 4)
POINTER(ivec4, 4)
POINTER(ivec3, 4)
POINTER(ivec2, 4)
POINTER(uvec4, 4)
POINTER(uvec3, 4)
POINTER(uvec2, 4)


struct Parameter
{
    GPUPtr data;
    uint stride[4];
    uint shape[4];
    GPUPtr grad_data;
};

#define DECLARE_TENSOR(dim) \
float_ptr param_buffer(in Parameter tensor, int index[dim]) { \
    uint element_offset = 0; \
    [[unroll]] \
    for (int d = 0; d < dim; d++) element_offset += tensor.stride[d] * index[d]; \
    return float_ptr(tensor.data + (element_offset << 2)); \
}\
float_ptr param_grad_buffer(in Parameter tensor, int index[dim]) { \
    uint element_offset = 0; \
    [[unroll]] \
    for (int d = 0; d < dim; d++) element_offset += tensor.stride[d] * index[d]; \
    return float_ptr(tensor.grad_data + (element_offset << 2)); \
}

DECLARE_TENSOR(1)

DECLARE_TENSOR(2)

DECLARE_TENSOR(3)

DECLARE_TENSOR(4)


float_ptr param_buffer(in Parameter tensor, int index) {
    return param_buffer(tensor, int[1](index));
}
float_ptr param_buffer(in Parameter tensor, ivec2 index) {
    return param_buffer(tensor, int[2](index.y, index.x));
}
float_ptr param_buffer(in Parameter tensor, ivec3 index) {
    return param_buffer(tensor, int[3](index.z, index.y, index.x));
}
float_ptr param_buffer(in Parameter tensor, ivec4 index) {
    return param_buffer(tensor, int[4](index.w, index.z, index.y, index.x));
}


float_ptr param_grad_buffer(in Parameter tensor, int index) {
    return param_grad_buffer(tensor, int[1](index));
}
float_ptr param_grad_buffer(in Parameter tensor, ivec2 index) {
    return param_grad_buffer(tensor, int[2](index.y, index.x));
}
float_ptr param_grad_buffer(in Parameter tensor, ivec3 index) {
    return param_grad_buffer(tensor, int[3](index.z, index.y, index.x));
}
float_ptr param_grad_buffer(in Parameter tensor, ivec4 index) {
    return param_grad_buffer(tensor, int[4](index.w, index.z, index.y, index.x));
}



float atomicAdd_f(float_ptr buf, int index, float value)
{
    #ifdef SUPPORTED_FLOAT_ATOM_ADD
    return atomicAdd(buf.data[index], value);
    #else
    uint_ptr buf_as_uint = uint_ptr(uint64_t(buf));
    uint old = buf_as_uint.data[index];
    uint assumed;
    do {
        assumed = old;
        old = atomicCompSwap(buf_as_uint.data[index], assumed, floatBitsToUint(value + uintBitsToFloat(assumed)));
    } while(assumed != old);
    return uintBitsToFloat(old);
    #endif
}


#define IS_NULL(x) (GPUPtr(x) == 0)

#endif