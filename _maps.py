from .rendering import Layout, lazy_constant, tensor, tensor_copy, \
    object_buffer, ObjectBufferAccessor, pipeline_compute, pipeline_raytracing, \
    compute_manager, graphics_manager, raytracing_manager, submit, wrap, GPUPtr, BufferUsage, MemoryLocation
from ._internal import device, get_seeds, __INCLUDE_PATH__
from ._functions import random_sphere_points, random_ids
from ._gmath import *
from typing import List, Tuple, Optional, Literal, Union
import torch
import os
from ._gmath import vec2, vec3, ivec3, ivec2
import numpy as np
import threading


__USE_VULKAN_DISPATCHER__ = True


def torch_fallback(enable=True):
    global __USE_VULKAN_DISPATCHER__
    current_dipatcher = __USE_VULKAN_DISPATCHER__
    __USE_VULKAN_DISPATCHER__ = not enable
    class _using_torch_context():
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            global __USE_VULKAN_DISPATCHER__
            __USE_VULKAN_DISPATCHER__ = current_dipatcher
    return _using_torch_context()


class DispatcherEngine:
    __REGISTERED_MAPS__ = []  # Each map type
    __REGISTERED_RAYCASTERS__ = []  # Each raycaster type  (FUTURE)
    __MAP_INSTANCES__ = {}  # All instances, from tuple with signature to codename.
    # __RAYCASTER_INSTANCES__ = { }  # All instances, from tuple with signature to codename.
    __RT_SUPER_KERNEL__ = ""  # Current engine code
    __CS_SUPER_KERNEL__ = ""  # Current engine code
    __INCLUDE_DIRS__ = []
    __DEFINED_STRUCTS__ = {}
    __ENGINE_OBJECTS__ = None  # Objects to dispatch map evaluation and raycasting

    __FW_RT_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for fw map evaluation
    __FW_CS_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for fw map evaluation
    __FW_DISPATCHER_CACHED_MAN__ = {}  # command buffers for dispatching fw map evaluations
    __FW_RAYCASTER_CACHED_MAN__ = {}  # command buffers for dispatching fw raycast evaluations
    __FW_CAPTURE_CACHED_MAN__ = {}  # command buffers for dispatching fw capture evaluations

    __BW_RT_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for bw map evaluation
    __BW_CS_ENGINE_PIPELINES__ = {}  # Pipelines and RT Programs for bw map evaluation
    __BW_DISPATCHER_CACHED_MAN__ = {}  # command buffers for dispatching bw map evaluations
    __BW_RAYCASTER_CACHED_MAN__ = {}  # command buffers for dispatching bw raycast evaluations
    __BW_CAPTURE_CACHED_MAN__ = {}  # command buffers for dispatching bw capture evaluations

    __LOCKER__ = threading.Lock()

    @classmethod
    def start(cls):
        _, inner_structs, _ = cls.create_code_for_struct_declaration(ParameterDescriptor)
        cls.__DEFINED_STRUCTS__.update(inner_structs)

    @classmethod
    def register_map(cls, map_type: 'MapMeta') -> int:
        code = len(cls.__REGISTERED_MAPS__) + 1
        cls.__REGISTERED_MAPS__.append(map_type)
        return code

    @classmethod
    def create_code_for_struct_declaration(cls, type_definition) -> Tuple[
        str, dict, list]:  # Code, new structures, sizes
        if type_definition == MapBase:
            raise Exception('Basic structs can not contain map references')
        if type_definition == torch.Tensor:
            return 'GPUPtr', {}, []
        if Layout.is_scalar_type(type_definition):
            return {
                       int: 'int',
                       float: 'float',
                       torch.int32: 'int',
                       torch.float32: 'float',
                   }[type_definition], {}, []
        if isinstance(type_definition, list):
            size = type_definition[0]
            t = type_definition[1]
            element_decl, inner_structures, element_sizes = cls.create_code_for_struct_declaration(t)
            return element_decl, inner_structures, [size] + element_sizes
        if isinstance(type_definition, dict):
            assert 'name' in type_definition, 'Basic structs must be named, include a key name: str'
            inner_structures = {}
            struct_code = f"struct {type_definition['name']} {{"
            for field_id, field_type in type_definition.items():
                if field_id != 'name':
                    t, field_inner_structures, sizes = cls.create_code_for_struct_declaration(field_type)
                    struct_code += t + " " + field_id + ''.join(f"[{size}]" for size in sizes) + '; \n'
                    inner_structures.update(field_inner_structures)
            struct_code += '};'
            inner_structures[type_definition['name']] = struct_code
            return type_definition['name'], inner_structures, []
        return type_definition.__name__, {}, []  # vec and mats

    @classmethod
    def create_code_for_map_declaration(cls, type_definition, field_value, allow_block: bool = False) -> Tuple[
        str, dict, list]:  # Code, new structures, sizes
        if type_definition == MapBase:
            return cls.register_instance(field_value.obj)[1], {}, []
        if type_definition == torch.Tensor:
            return 'GPUPtr', {}, []
        if Layout.is_scalar_type(type_definition):
            return {
                       int: 'int',
                       float: 'float',
                       torch.int32: 'int',
                       torch.float32: 'float',
                   }[type_definition], {}, []
        if isinstance(type_definition, list):
            size = type_definition[0]
            t = type_definition[1]
            field_value: ObjectBufferAccessor
            element_decl, inner_structures, element_sizes = cls.create_code_for_map_declaration(t, field_value[0])
            return element_decl, inner_structures, [size] + element_sizes
        if isinstance(type_definition, dict):
            inner_structures = {}
            if 'name' in type_definition:  # external struct
                struct_code = f"struct {type_definition['name']} {{"
                for field_id, field_type in type_definition.items():
                    if field_id != 'name':
                        t, field_inner_structures, sizes = cls.create_code_for_map_declaration(field_type,
                                                                                               getattr(field_value,
                                                                                                       field_id))
                        struct_code += t + " " + field_id + ''.join(f"[{size}]" for size in sizes) + '; \n'
                        inner_structures.update(field_inner_structures)
                struct_code += '};'
                inner_structures[type_definition['name']] = struct_code
                return type_definition['name'], inner_structures, []
            else:  # block
                assert allow_block, 'Can not create a nested block. Add a name attribute to the dictionary to make it a struct'
                code = "{"
                for field_id, field_type in type_definition.items():
                    t, field_inner_structures, sizes = cls.create_code_for_map_declaration(field_type,
                                                                                           getattr(field_value,
                                                                                                   field_id))
                    code += t + " " + field_id + ''.join(f"[{size}]" for size in sizes) + '; \n'
                    inner_structures.update(field_inner_structures)
                code += '}'
                return code, inner_structures, []
        return type_definition.__name__, {}, []  # vec and mats

    @classmethod
    def append_map_instance_source_code(cls, map: 'MapBase', codename: str):
        code = ""
        map_object_parameters_code, external_structs, _ = cls.create_code_for_map_declaration(map.map_object_definition,
                                                                                              map._rdv_map_buffer_accessor,
                                                                                              allow_block=True)
        for struct_name, struct_code in external_structs.items():
            if struct_name in cls.__DEFINED_STRUCTS__:
                assert cls.__DEFINED_STRUCTS__[
                           struct_name] == struct_code, f'A different body was already defined for {struct_name}'
            else:
                code += struct_code + "\n"
                cls.__DEFINED_STRUCTS__[struct_name] = struct_code
        # Add buffer_reference definition with codename and map object layout
        code += f"""
layout(buffer_reference, scalar, buffer_reference_align=4) buffer buffer_{codename} {map_object_parameters_code};
struct {codename} {{ buffer_{codename} data; }};
"""
        for g, v in map.generics.items():
            code += f"#define {g} {v} \n"
        code += f"#define map_object in {codename} object \n"
        code += f"#define parameters object.data \n"

        code += map.map_source_code + "\n"

        if map.non_differentiable:
            code += """
void backward (map_object, float _input[INPUT_DIM], float _output_grad[OUTPUT_DIM], inout float _input_grad[INPUT_DIM]) {  }
"""
        code += """
void backward (map_object, float _input[INPUT_DIM], float _output_grad[OUTPUT_DIM]) {
    float _input_grad[INPUT_DIM];
    backward(object, _input, _output_grad, _input_grad);  
}
        """

        code += f"#undef map_object\n"
        code += f"#undef parameters\n"

        for g in map.generics:
            code += f"#undef {g}\n"

        cls.__RT_SUPER_KERNEL__ += code
        if not map.requires_raytracing:
            cls.__CS_SUPER_KERNEL__ += code
        cls.__INCLUDE_DIRS__.extend(map.include_dirs)

    @classmethod
    def register_instance(cls, map: 'MapBase') -> Tuple[int, str]:  # new or existing instance id for the object
        signature = map.signature
        if signature not in cls.__MAP_INSTANCES__:
            instance_id = len(cls.__MAP_INSTANCES__) + 1
            codename = 'rdv_map_' + str(instance_id)
            cls.append_map_instance_source_code(map, codename)
            cls.__MAP_INSTANCES__[signature] = (instance_id, codename)
        return cls.__MAP_INSTANCES__[signature]

    @classmethod
    def register_raycaster(cls, raycaster: 'RaycasterMeta') -> int:
        code = len(cls.__REGISTERED_RAYCASTERS__)
        cls.__REGISTERED_RAYCASTERS__.append(raycaster)
        return code

    @classmethod
    def ensure_engine_objects(cls):
        if cls.__ENGINE_OBJECTS__ is not None:
            return

        map_fw_eval = object_buffer(element_description=Layout.create_structure('std430',
                                                                                main_map=torch.int64,
                                                                                input=torch.int64,
                                                                                output=torch.int64,
                                                                                seeds=ivec4,
                                                                                start_index=int,
                                                                                total_threads=int
                                                                                ))

        map_bw_eval = object_buffer(element_description=Layout.create_structure('std430',
                                                                                main_map=torch.int64,
                                                                                input=torch.int64,
                                                                                output_grad=torch.int64,
                                                                                seeds=ivec4,
                                                                                start_index=int,
                                                                                total_threads=int
                                                                                ))

        capture_fw_eval = object_buffer(element_description=Layout.create_structure('std430',
                                                                                    capture_sensor=torch.int64,
                                                                                    main_map=torch.int64,
                                                                                    sensors=torch.int64,
                                                                                    tensor=torch.int64,
                                                                                    seeds=ivec4,
                                                                                    sensor_shape=[4, int],
                                                                                    start_index=int,
                                                                                    total_threads=int,
                                                                                    samples=int,
                                                                                    ))

        capture_bw_eval = object_buffer(element_description=Layout.create_structure('std430',
                                                                                    capture_sensor=torch.int64,
                                                                                    main_map=torch.int64,
                                                                                    sensors=torch.int64,
                                                                                    tensor_grad=torch.int64,
                                                                                    seeds=ivec4,
                                                                                    sensor_shape=[4, int],
                                                                                    start_index=int,
                                                                                    total_threads=int,
                                                                                    samples=int,
                                                                                    ))
        cls.__ENGINE_OBJECTS__ = {
            'map_fw_eval': map_fw_eval,
            'map_bw_eval': map_bw_eval,
            'capture_fw_eval': capture_fw_eval,
            'capture_bw_eval': capture_bw_eval,
        }

    @classmethod
    def build_map_fw_eval_objects(cls, map: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval map
        """
        if map.signature in cls.__FW_CS_ENGINE_PIPELINES__:
            return cls.__FW_CS_ENGINE_PIPELINES__[map.signature]

        full_code = """
#version 460
#extension GL_GOOGLE_include_directive : require
#include "common.h"
layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        """ + cls.__CS_SUPER_KERNEL__ + f"""

layout(set = 0, std430, binding = 0) uniform RayGenMainDispatching {{
    {cls.register_instance(map)[1]} main_map; // Map model to execute
    GPUPtr input_tensor_ptr; // Input tensor (forward and backward stage)
    GPUPtr output_tensor_ptr; // Output tensor (forward stage)
    uvec4 seeds; // seeds for the batch randoms
    int start_index;
    int total_threads;
}};        

layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_data {{ float data [{map.input_dim}]; }};
layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{map.output_dim}]; }};

void main()
{{
    int index = int(gl_GlobalInvocationID.x) + start_index;
    if (index >= total_threads) return;

    uvec4 current_seeds = seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
    set_seed(current_seeds);
    random();
    random();
    random();

    int input_dim = {map.input_dim};
    int output_dim = {map.output_dim};
    rdv_input_data input_buf = rdv_input_data(input_tensor_ptr + index * input_dim * 4);
    rdv_output_data output_buf = rdv_output_data(output_tensor_ptr + index * output_dim * 4);
    forward(main_map, input_buf.data, output_buf.data);
}}
        """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = pipeline_compute()
        pipeline.bind_uniform(0, cls.__ENGINE_OBJECTS__['map_fw_eval'])
        pipeline.load_shader_from_source(full_code, include_dirs=[__INCLUDE_PATH__] + cls.__INCLUDE_DIRS__)
        pipeline.close()

        cls.__FW_CS_ENGINE_PIPELINES__[map.signature] = pipeline

        return pipeline

    @classmethod
    def build_map_bw_eval_objects(cls, map: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval map
        """
        if map.signature in cls.__BW_CS_ENGINE_PIPELINES__:
            return cls.__BW_CS_ENGINE_PIPELINES__[map.signature]

        full_code = """
    #version 460
    #extension GL_GOOGLE_include_directive : require
    #include "common.h"
    layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

            """ + cls.__CS_SUPER_KERNEL__ + f"""

    layout(set = 0, std430, binding = 0) uniform BackwardMapEval {{
        {cls.register_instance(map)[1]} main_map; // Map model to execute
        GPUPtr input_tensor_ptr; // Input tensor (forward and backward stage)
        GPUPtr output_tensor_grad_ptr; // Output tensor (backward stage)
        uvec4 seeds; // seeds for the batch randoms
        int start_index;
        int total_threads;
    }};        

    layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_input_data {{ float data [{map.input_dim}]; }};
    layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{map.output_dim}]; }};

    void main()
    {{
        int index = int(gl_GlobalInvocationID.x) + start_index;
        if (index >= total_threads) return;

        uvec4 current_seeds = seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
        set_seed(current_seeds);
        random();
        random();
        random();

        int input_dim = {map.input_dim};
        int output_dim = {map.output_dim};
        rdv_input_data input_buf = rdv_input_data(input_tensor_ptr + index * input_dim * 4);
        rdv_output_data output_grad_buf = rdv_output_data(output_tensor_grad_ptr + index * output_dim * 4);
        backward(main_map, input_buf.data, output_grad_buf.data);
    }}
            """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = pipeline_compute()
        pipeline.bind_uniform(0, cls.__ENGINE_OBJECTS__['map_bw_eval'])
        pipeline.load_shader_from_source(full_code, include_dirs=[__INCLUDE_PATH__] + cls.__INCLUDE_DIRS__)
        pipeline.close()

        cls.__BW_CS_ENGINE_PIPELINES__[map.signature] = pipeline

        return pipeline

    @classmethod
    def build_capture_fw_eval_objects(cls, capture_object: 'SensorsBase', field: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval the specific capture of map
        """
        capture_signature = (capture_object.signature, field.signature)
        if capture_signature in cls.__FW_CS_ENGINE_PIPELINES__:
            return cls.__FW_CS_ENGINE_PIPELINES__[capture_signature]

        full_code = """
        #version 460
        #extension GL_GOOGLE_include_directive : require
        #include "common.h"
        layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

                """ + cls.__CS_SUPER_KERNEL__ + f"""

        layout(set = 0, std430, binding = 0) uniform RayGenMainDispatching {{
            {cls.register_instance(capture_object)[1]} capture_object; // Map model to execute
            {cls.register_instance(field)[1]} main_map; // Map model to execute
            GPUPtr sensors_ptr; // Input tensor for sensors batch
            GPUPtr output_tensor_ptr; // Output tensor (forward stage)
            uvec4 seeds; // seeds for the batch randoms
            int sensor_shape[4]; // up to 4 lengths for each index dimension
            int start_index;
            int total_threads;
            int samples;
        }};        

        layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{field.output_dim}]; }};

        void main()
        {{
            int index = int(gl_GlobalInvocationID.x) + start_index;
            if (index >= total_threads) return;

            uvec4 current_seeds = seeds + uvec4(0x23F1,0x3137,129,index + 129) ;//^ uvec4(int(cos(index)*1000000), index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));

            //uvec4 current_seeds = seeds ^ uvec4(index * 78182311, index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
            set_seed(current_seeds);
            advance_random();
            advance_random();
            advance_random();
            advance_random();
            uvec4 new_seed = floatBitsToUint(vec4(random(), random(), random(), random()));
            set_seed(new_seed);

            float indices[{capture_object.input_dim}];

            if (sensors_ptr == 0) {{
                int current_index_component = index;
                [[unroll]]
                for (int i={capture_object.input_dim} - 1; i>=0; i--)
                {{
                    int d = sensor_shape[i];
                    indices[i] = intBitsToFloat(current_index_component % d);
                    current_index_component /= d;
                }}
            }}
            else
            {{
                int_ptr sensors_buf = int_ptr(sensors_ptr + 8 * {capture_object.input_dim} * index);
                [[unroll]]
                for (int i=0; i<{capture_object.input_dim}; i++)
                    indices[i] = intBitsToFloat(sensors_buf.data[i*2]);
            }}

            if (samples == 1) {{
                float sensor_position[{field.input_dim}];
                forward(capture_object, indices, sensor_position);

                int output_dim = {field.output_dim};
                rdv_output_data output_buf = rdv_output_data(output_tensor_ptr + index * output_dim * 4);
                forward(main_map, sensor_position, output_buf.data);
            }}
            else {{
                int output_dim = {field.output_dim};
                rdv_output_data output_buf = rdv_output_data(output_tensor_ptr + index * output_dim * 4);
                [[unroll]]
                for (int i=0; i<output_dim; i++)
                    output_buf.data[i] = 0.0;
                for (int s = 0; s < samples; s ++)
                {{
                    float sensor_position[{field.input_dim}];
                    forward(capture_object, indices, sensor_position);
                    float temp_output [{field.output_dim}];
                    forward(main_map, sensor_position, temp_output);
                    [[unroll]]
                    for (int i=0; i<output_dim; i++)
                        output_buf.data[i] += temp_output[i];
                }}
                [[unroll]]
                for (int i=0; i<output_dim; i++)
                    output_buf.data[i] /= samples;
            }}
        }}
                """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = pipeline_compute()
        pipeline.bind_uniform(0, cls.__ENGINE_OBJECTS__['capture_fw_eval'])
        pipeline.load_shader_from_source(full_code, include_dirs=[__INCLUDE_PATH__]+cls.__INCLUDE_DIRS__)
        pipeline.close()

        cls.__FW_CS_ENGINE_PIPELINES__[capture_signature] = pipeline

        return pipeline

    @classmethod
    def build_capture_bw_eval_objects(cls, capture_object: 'SensorsBase', field: 'MapBase'):
        """
        Creates a raytracing or compute pipeline with the required buffers to eval the specific capture of map
        """
        capture_signature = (capture_object.signature, field.signature)
        if capture_signature in cls.__BW_CS_ENGINE_PIPELINES__:
            return cls.__BW_CS_ENGINE_PIPELINES__[capture_signature]

        full_code = """
        #version 460
        #extension GL_GOOGLE_include_directive : require
        #include "common.h"
        layout (local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

                """ + cls.__CS_SUPER_KERNEL__ + f"""

        layout(set = 0, std430, binding = 0) uniform BackwardCaptureEval {{
            {cls.register_instance(capture_object)[1]} capture_object; // Map model to execute
            {cls.register_instance(field)[1]} main_map; // Map model to execute
            GPUPtr sensors_ptr; // Input tensor for sensors batch
            GPUPtr output_grad_tensor_ptr; // Output gradients tensor (backward stage)
            uvec4 seeds; // seeds for the batch randoms
            int sensor_shape[4]; // up to 4 lengths for each index dimension
            int start_index;
            int total_threads;
            int samples;
        }};        

        layout(buffer_reference, scalar, buffer_reference_align=4) buffer rdv_output_data {{ float data [{field.output_dim}]; }};

        void main()
        {{
            int index = int(gl_GlobalInvocationID.x) + start_index;
            if (index >= total_threads) return;

            uvec4 current_seeds = seeds + uvec4(0x23F1,0x3137,129,index + 129) ;//^ uvec4(int(cos(index)*1000000), index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));

            //uvec4 current_seeds = seeds ^ uvec4(index * 78182311, index ^ 1231231, index + 1234122, index + seeds.w * 100202021);//seeds ^ uvec4(index ^ 17, index * 123111171, index + 11, index ^ (seeds.x + 13 * seeds.y));
            set_seed(current_seeds);
            advance_random();
            advance_random();
            advance_random();
            advance_random();
            uvec4 new_seed = floatBitsToUint(vec4(random(), random(), random(), random()));
            set_seed(new_seed);

            float indices[{capture_object.input_dim}];
            if (sensors_ptr == 0) {{
                int current_index_component = index;
                [[unroll]]
                for (int i={capture_object.input_dim} - 1; i>=0; i--)
                {{
                    int d = sensor_shape[i];
                    indices[i] = intBitsToFloat(current_index_component % d);
                    current_index_component /= d;
                }}
            }}
            else
            {{
                int_ptr sensors_buf = int_ptr(sensors_ptr + 8 * {capture_object.input_dim} * index);
                [[unroll]]
                for (int i=0; i<{capture_object.input_dim}; i++)
                    indices[i] = intBitsToFloat(sensors_buf.data[i*2]);
            }}

            float dL_dp [{field.input_dim}];
            [[unroll]] for (int i = 0; i < {field.input_dim}; i++) dL_dp[i] = 0.0;
            
            int output_dim = {field.output_dim};
            rdv_output_data output_grad_buf = rdv_output_data(output_grad_tensor_ptr + index * output_dim * 4);
            float output_grad[{field.output_dim}];
            [[unroll]] for (int i=0; i< {field.output_dim}; i++) output_grad[i] = output_grad_buf.data[i] / samples;
            
            float sensor_position[{field.input_dim}];

            [[unroll]]
            for (int s = 0; s < samples; s++)
            {{
                forward(capture_object, indices, sensor_position);
                backward(main_map, sensor_position, output_grad, dL_dp);
            }}
            backward(capture_object, indices, dL_dp); // TODO: Check ways to update sensor parameters from ray differentials
        }}
                """

        cls.ensure_engine_objects()
        # Build pipeline for forward map evaluation
        pipeline = pipeline_compute()
        pipeline.bind_uniform(0, cls.__ENGINE_OBJECTS__['capture_bw_eval'])
        pipeline.load_shader_from_source(full_code, include_dirs=[__INCLUDE_PATH__] + cls.__INCLUDE_DIRS__)
        pipeline.close()

        cls.__BW_CS_ENGINE_PIPELINES__[capture_signature] = pipeline

        return pipeline

    @classmethod
    def eval_capture_forward(cls, capture_object: 'SensorsBase', field: 'MapBase',
                             sensors: Optional[torch.Tensor] = None, batch_size: Optional[int] = None,
                             fw_samples: int = 1) -> torch.Tensor:
        if sensors is not None:
            total_threads = sensors.numel() // sensors.shape[-1]
        else:
            total_threads = math.prod(capture_object.index_shape[
                                      :capture_object.input_dim]).item()  # capture_object.screen_width * capture_object.screen_height

        if batch_size is None:
            batch_size = total_threads

        cache_key = (batch_size, capture_object.signature, field.signature)

        # create man if not cached
        if cache_key not in cls.__FW_CAPTURE_CACHED_MAN__:
            pipeline = cls.build_capture_fw_eval_objects(capture_object, field)
            man = compute_manager()
            man.set_pipeline(pipeline)
            man.dispatch_threads_1D(batch_size, group_size_x=32)
            man.freeze()
            cls.__FW_CAPTURE_CACHED_MAN__[cache_key] = man

        man = cls.__FW_CAPTURE_CACHED_MAN__[cache_key]

        # assert input.shape[-1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'

        if sensors is not None:
            output = tensor(*sensors.shape[:-1], field.output_dim, dtype=torch.float)
        else:
            output = tensor(*capture_object.index_shape[:capture_object.input_dim], field.output_dim, dtype=torch.float)

        capture_object._pre_eval(False)
        field._pre_eval(False)

        output_ptr = wrap(output, 'out')

        with cls.__ENGINE_OBJECTS__['capture_fw_eval'] as b:
            b.capture_sensor = wrap(capture_object)
            b.main_map = wrap(field)
            b.sensors = wrap(sensors)
            b.tensor = output_ptr
            b.seeds[:] = get_seeds()
            shape = b.sensor_shape
            shape[0] = capture_object.index_shape[0].item()
            shape[1] = capture_object.index_shape[1].item()
            shape[2] = capture_object.index_shape[2].item()
            shape[3] = capture_object.index_shape[3].item()
            b.start_index = 0
            b.total_threads = total_threads
            b.samples = fw_samples

        for batch in range((total_threads + batch_size - 1) // batch_size):
            with cls.__ENGINE_OBJECTS__['capture_fw_eval'] as b:
                b.start_index = batch * batch_size
            submit(man)

        # output_ptr.mark_as_dirty()

        capture_object._pos_eval(False)
        field._pos_eval(False)

        # if sensors is None:
        #     output = output.view(*capture_object.index_shape[:capture_object.input_dim],-1)

        return output

    @classmethod
    def eval_capture_backward(cls, capture_object: 'SensorsBase', field: 'MapBase', output_grad: torch.Tensor,
                              sensors: Optional[torch.Tensor] = None, batch_size: Optional[int] = None,
                              bw_samples: int = 1):
        with cls.__LOCKER__:
            if sensors is not None:
                total_threads = sensors.numel() // sensors.shape[-1]
            else:
                total_threads = math.prod(capture_object.index_shape[
                                          :capture_object.input_dim]).item()  # capture_object.screen_width * capture_object.screen_height

            if batch_size is None:
                batch_size = total_threads

            cache_key = (batch_size, capture_object.signature, field.signature)

            # create man if not cached
            if cache_key not in cls.__BW_CAPTURE_CACHED_MAN__:
                pipeline = cls.build_capture_bw_eval_objects(capture_object, field)
                man = compute_manager()
                man.set_pipeline(pipeline)
                man.dispatch_threads_1D(batch_size, group_size_x=32)
                man.freeze()
                cls.__BW_CAPTURE_CACHED_MAN__[cache_key] = man

            man = cls.__BW_CAPTURE_CACHED_MAN__[cache_key]

            # assert input.shape[-1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'

            capture_object._pre_eval(True)
            field._pre_eval(True)

            with cls.__ENGINE_OBJECTS__['capture_bw_eval'] as b:
                b.capture_sensor = wrap(capture_object)
                b.main_map = wrap(field)
                b.sensors = wrap(sensors)
                b.tensor_grad = wrap(output_grad)
                b.seeds[:] = get_seeds()
                shape = b.sensor_shape
                shape[0] = capture_object.index_shape[0].item()
                shape[1] = capture_object.index_shape[1].item()
                shape[2] = capture_object.index_shape[2].item()
                shape[3] = capture_object.index_shape[3].item()
                b.start_index = 0
                b.total_threads = total_threads
                b.samples = bw_samples

            for batch in range((total_threads + batch_size - 1) // batch_size):
                with cls.__ENGINE_OBJECTS__['capture_bw_eval'] as b:
                    b.start_index = batch * batch_size
                submit(man)

            capture_object._pos_eval(True)
            field._pos_eval(True)

    @classmethod
    def eval_map_forward(cls, map_object: 'MapBase', input: torch.Tensor) -> torch.Tensor:
        total_threads = math.prod(input.shape[:-1])

        cache_key = (total_threads, map_object.signature)

        # create man if not cached
        if cache_key not in cls.__FW_DISPATCHER_CACHED_MAN__:
            pipeline = cls.build_map_fw_eval_objects(map_object)
            man = compute_manager()
            man.set_pipeline(pipeline)
            man.dispatch_threads_1D(total_threads, group_size_x=32)
            man.freeze()
            cls.__FW_DISPATCHER_CACHED_MAN__[cache_key] = man

        man = cls.__FW_DISPATCHER_CACHED_MAN__[cache_key]

        assert input.shape[
                   -1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'
        output = tensor(*input.shape[:-1], map_object.output_dim, dtype=torch.float)

        map_object._pre_eval(False)

        with cls.__ENGINE_OBJECTS__['map_fw_eval'] as b:
            b.main_map = wrap(map_object)
            b.input = wrap(input)
            b.output = wrap(output, 'out')
            b.seeds[:] = get_seeds()
            b.start_index = 0
            b.total_threads = total_threads

        submit(man)

        map_object._pos_eval(False)

        return output

    @classmethod
    def eval_map_backward(cls, map_object: 'MapBase', input: torch.Tensor, output_grad: torch.Tensor):
        with cls.__LOCKER__:
            total_threads = math.prod(input.shape[:-1])

            cache_key = (total_threads, map_object.signature)

            # create man if not cached
            if cache_key not in cls.__BW_DISPATCHER_CACHED_MAN__:
                pipeline = cls.build_map_bw_eval_objects(map_object)
                man = compute_manager()
                man.set_pipeline(pipeline)
                man.dispatch_threads_1D(total_threads, group_size_x=32)
                man.freeze()
                cls.__BW_DISPATCHER_CACHED_MAN__[cache_key] = man

            man = cls.__BW_DISPATCHER_CACHED_MAN__[cache_key]

            assert input.shape[
                       -1] == map_object.input_dim, f'Wrong last dimension for the input tensor, must be {map_object.input_dim}'
            assert output_grad.shape[
                       -1] == map_object.output_dim, f'Wrong last dimension for the output_grad tensor, must be {map_object.output_dim}'

            map_object._pre_eval(True)

            with cls.__ENGINE_OBJECTS__['map_bw_eval'] as b:
                b.main_map = wrap(map_object)
                b.input = wrap(input)
                b.output_grad = wrap(output_grad)
                b.seeds[:] = get_seeds()
                b.start_index = 0
                b.total_threads = total_threads

            submit(man)

            map_object._pos_eval(True)


def start_engine():
    if torch.cuda.is_available():
        torch.cuda.init()
    DispatcherEngine.start()


ParameterDescriptor = dict(
    name='Parameter',
    data=torch.Tensor,
    stride=[4, int],
    shape=[4, int],
    grad_data=torch.Tensor
)


def parameter(p: Union[torch.Tensor, torch.nn.Parameter]):
    if isinstance(p, torch.nn.Parameter):
        return p
    return torch.nn.Parameter(p, requires_grad=p.requires_grad)


def bind_parameter(field: ObjectBufferAccessor, t: torch.Tensor):
    field.data = wrap(t, 'in')
    for i, s in enumerate(t.shape):
        field.stride[i] = t.stride(i)
        field.shape[i] = s


def bind_parameter_grad(field: ObjectBufferAccessor):
    t: torch.Tensor = field.data.obj
    if t.requires_grad:
        if t.grad is None:
            t.grad = tensor(*t.shape, dtype=t.dtype).zero_()
        field.grad_data = wrap(t.grad, 'inout')


class MapMeta(type):
    def __new__(cls, name, bases, dct):
        ext_class = super().__new__(cls, name, bases, dct)
        assert '__extension_info__' in dct, 'Extension maps requires a dict __extension_info__ with path, parameters, [optional] nodiff'
        extension_info = dct['__extension_info__']
        if extension_info is not None:  # is not an abstract node
            extension_path = extension_info.get('path', None)
            extension_code = extension_info.get('code', None)
            extension_generics = extension_info.get('generics', {})
            assert extension_path is None or isinstance(extension_path, str) and os.path.isfile(
                extension_path), 'path must be a valid file path str'
            include_dirs = extension_info.get('include_dirs', [])
            assert (extension_path is None) != (extension_code is None), 'Either path or code must be provided'
            if extension_path is not None:
                include_dirs.append(os.path.dirname(extension_path))
                extension_code = f"#include \"{os.path.basename(extension_path)}\"\n"
                # with open(extension_path) as f:
                #     extension_code = f.readlines()
            parameters = extension_info.get('parameters', {})
            if parameters is None or len(parameters) == 0:
                parameters = dict(foo=int)
            non_differentiable = extension_info.get('nodiff', False)
            use_raycast = extension_info.get('use_raycast', False)

            def from_type_2_layout_description(p):
                if p == MapBase:
                    return torch.int64
                if p == torch.Tensor:
                    return torch.int64
                if isinstance(p, list):
                    return [p[0], from_type_2_layout_description(p[1])]
                if isinstance(p, dict):
                    return {k: from_type_2_layout_description(v) for k, v in p.items() if k != 'name'}
                return p

            parameters_layout = Layout.create(from_type_2_layout_description(parameters), mode='scalar')
            ext_class.default_generics = extension_generics
            ext_class.map_object_layout = parameters_layout
            ext_class.map_object_definition = parameters
            ext_class.map_source_code = extension_code
            ext_class.non_differentiable = non_differentiable
            ext_class.use_raycast = use_raycast
            ext_class.include_dirs = include_dirs
            ext_class.map_code = DispatcherEngine.register_map(ext_class)
        return ext_class

    def __call__(self, *args, **kwargs):
        map_instance: MapBase = super(MapMeta, self).__call__(*args, **kwargs)
        map_instance._freeze_submodules()
        map_id, map_codename = DispatcherEngine.register_instance(map_instance)
        object.__setattr__(map_instance, 'map_id', map_id)
        return map_instance


class MapBase(torch.nn.Module, metaclass=MapMeta):
    __extension_info__ = None  # none extension info marks the node as abstract
    __bindable__ = None
    map_object_layout: Layout = None

    def __init__(self, **generics):
        map_buffer = object_buffer(element_description=self.map_object_layout, usage=BufferUsage.STAGING,
                                   memory=MemoryLocation.CPU)
        object.__setattr__(self, '__bindable__', map_buffer)
        object.__setattr__(self, '_rdv_map_buffer', map_buffer)
        object.__setattr__(self, '_rdv_map_buffer_accessor', map_buffer.accessor)
        object.__setattr__(self, '_rdv_trigger_bw', torch.tensor([0.0], requires_grad=True))
        object.__setattr__(self, '_rdv_no_trigger_bw', torch.tensor([0.0], requires_grad=False))
        object.__setattr__(self, 'generics', {**self.default_generics, **generics})
        for k, (offset, field_type) in self.map_object_layout.fields_layout.items():
            if field_type.is_array:  # lists
                object.__setattr__(self, k, getattr(map_buffer.accessor, k))
        super(MapBase, self).__init__()

    @lazy_constant
    def input_dim(self):
        return self.generics['INPUT_DIM']

    @lazy_constant
    def output_dim(self):
        return self.generics['OUTPUT_DIM']

    def get_maximum(self) -> torch.Tensor:
        raise NotImplementedError()

    def support_maximum_query(self) -> bool:
        return False

    # def __getattr__(self, item):
    #     a : ObjectBufferAccessor = object.__getattribute__(self, '_rdv_map_buffer_accessor')
    #     if item in a._rdv_fields:
    #         return getattr(a, item)
    #     return super(MapBase, self).__getattr__(item)

    def __setattr__(self, key, value):
        a: ObjectBufferAccessor = self._rdv_map_buffer_accessor
        if key in a._rdv_fields:
            if a._rdv_layout.fields_layout[key][1].scalar_format == 'Q':
                setattr(a, key, wrap(value))
            elif key in self._parameters or isinstance(value, torch.nn.Parameter):
                bind_parameter(getattr(a, key), value)
            else:
                setattr(a, key, value)
        super(MapBase, self).__setattr__(key, value)

    def _freeze_submodules(self):
        requires_raytracing = self.use_raycast
        submodules = []

        def collect_submodules(field_name, t, v):
            if t == MapBase:
                assert v is not None, f'Expected module {field_name} to be bound'
                submodules.append(v.obj)
                return
            if isinstance(t, list):
                size = t[0]
                element_type = t[1]
                collect_submodules(field_name, element_type, v[
                    0])  # only collect from zero index, TODO: CHECK all array derivations have the same signature!
            if isinstance(t, dict):
                for field_name, field_type in t.items():
                    if field_name != 'name':
                        collect_submodules(field_name, field_type, getattr(v, field_name))

        collect_submodules('', self.map_object_definition, self._rdv_map_buffer_accessor)

        for sub_object in submodules:
            requires_raytracing |= sub_object.requires_raytracing  # append if any use ray or random

        object.__setattr__(self, 'requires_raytracing', requires_raytracing)
        object.__setattr__(self, 'signature', (
        self.map_code, *[0 if s is None else DispatcherEngine.register_instance(s)[0] for s in submodules],
        *[v for k, v in self.generics.items()]))

    def _pre_eval(self, include_grads: bool = False):
        if include_grads:
            for k,v in self._parameters.items():
                if v.requires_grad:
                    bind_parameter_grad(getattr(self._rdv_map_buffer_accessor, k))
        for k, m in self._modules.items():
            if isinstance(m, MapBase):
                m._pre_eval(include_grads)
        for r in self._rdv_map_buffer_accessor.references():
            if r is not None:
                r.flush()
        self._rdv_map_buffer.update_gpu()

    def _pos_eval(self, include_grads: bool = False):
        for k, m in self._modules.items():
            if isinstance(m, MapBase):
                m._pos_eval(include_grads)
        for r in self._rdv_map_buffer_accessor.references():
            if r is not None:
                r.mark_as_dirty()
                r.invalidate()

    def forward_torch(self, *args):
        raise Exception(f"Not implemented torch engine in type {type(self)}")

    def forward(self, *args):
        # Use generic evaluator
        if not __USE_VULKAN_DISPATCHER__:
            return self.forward_torch(*args)
        else:
            trigger_bw = self._rdv_trigger_bw if any(True for _ in self.parameters()) else self._rdv_no_trigger_bw
            return AutogradMapFunction.apply(*args, trigger_bw, self)

    def after(self, prev_map: 'MapBase') -> 'MapBase':
        return Composition(prev_map, self)

    def then(self, next_map: 'MapBase') -> 'MapBase':
        return Composition(self, next_map)

    def promote(self, output_dim: int) -> 'MapBase':
        return PromoteMap(self, output_dim)

    def like_this(self, o):
        if isinstance(o, int) or isinstance(o, float):
            t = tensor(self.output_dim, dtype=torch.float32)
            t[:] = o
            return ConstantMap(self.input_dim, t)
        if isinstance(o, torch.Tensor):
            o = ConstantMap(self.input_dim, o)
        if isinstance(o, MapBase):
            if o.output_dim == self.output_dim:
                return o
            return o.promote(self.output_dim)
        raise Exception(f'Can not cast type {type(o)} to this map')

    @staticmethod
    def make_match(v1, v2):
        if not isinstance(v1, MapBase):
            assert isinstance(v2, MapBase)
            v2, v1 = MapBase.make_match(v2, v1)
            return v1, v2
        if not isinstance(v2, MapBase) or v2.output_dim < v1.output_dim:
            return v1, v1.like_this(v2)
        return v2.like_this(v1), v2

    def __add__(self, other):
        return AdditionMap(*MapBase.make_match(self, other))

    def __radd__(self, other):
        return AdditionMap(*MapBase.make_match(self, other))

    def __sub__(self, other):
        return SubtractionMap(*MapBase.make_match(self, other))

    def __rsub__(self, other):
        return SubtractionMap(*MapBase.make_match(other, self))

    def __mul__(self, other):
        return MultiplicationMap(*MapBase.make_match(self, other))

    def __rmul__(self, other):
        return MultiplicationMap(*MapBase.make_match(self, other))

    def __truediv__(self, other):
        return DivisionMap(*MapBase.make_match(self, other))

    def __rtruediv__(self, other):
        return DivisionMap(*MapBase.make_match(other, self))

    def __getitem__(self, item):
        if isinstance(item, int):
            return IndexMap(self, [item])
        if isinstance(item, tuple) or isinstance(item, list):
            return IndexMap(self, list(item))
        if isinstance(item, slice):
            indices = [i for i in range(self.output_dim)]
            return IndexMap(self, indices[item])
        raise Exception(f"Not supported index/slice object {type(item)}")

    def __or__(self, other):
        return ConcatMap(self, other)


class AutogradMapFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        input_tensor, triggering, map_object = args
        ctx.save_for_backward(input_tensor)  # properly save tensors for backward
        ctx.map_object = map_object
        return DispatcherEngine.eval_map_forward(map_object, input_tensor)

    @staticmethod
    def backward(ctx, *args):
        output_grad, = args
        input_tensor, = ctx.saved_tensors  # Just check for inplace operations in input tensors
        map_object = ctx.map_object
        DispatcherEngine.eval_map_backward(map_object, input_tensor, output_grad)
        return (None, None, None)  # append None to refer to renderer object passed in forward


class AutogradCaptureFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        sensors, triggering, batch_size, fw_samples, bw_samples, field, capture_object = args
        ctx.field = field
        ctx.sensors = sensors
        ctx.capture_object = capture_object
        ctx.batch_size = batch_size
        ctx.bw_samples = bw_samples
        return DispatcherEngine.eval_capture_forward(capture_object, field, sensors, batch_size, fw_samples)

    @staticmethod
    def backward(ctx, *args):
        output_grad, = args
        capture = ctx.capture_object
        sensors = ctx.sensors
        field = ctx.field
        batch_size = ctx.batch_size
        bw_samples = ctx.bw_samples
        DispatcherEngine.eval_capture_backward(capture, field, output_grad, sensors, batch_size, bw_samples)
        # print(f"[DEBUG] Backward grads from renderer {grad_inputs[0].mean()}")
        # assert grad_inputs[0] is None or torch.isnan(grad_inputs[0]).sum() == 0, "error in generated grads."
        return (None, None, None, None, None, None, None)  # append None to refer to renderer object passed in forward


class Identity(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/maps/identity.h"
    )

    def __init__(self, dimension: int):
        super(Identity, self).__init__(INPUT_DIM=dimension, OUTPUT_DIM=dimension)

    def forward_torch(self, *args):
        x, = args
        return x


class SensorsBase(MapBase):
    __extension_info__ = None  # Abstract node

    def __init__(self, index_shape: List[int], **generics):
        super(SensorsBase, self).__init__(INPUT_DIM=len(index_shape), **generics)
        object.__setattr__(self, 'identity', Identity(self.output_dim))
        object.__setattr__(self, 'index_shape', torch.tensor(index_shape + ([0] * (4 - len(index_shape)))))
        object.__setattr__(self, '_rdv_trigger_bw', torch.tensor([0.0], requires_grad=True))
        object.__setattr__(self, '_rdv_no_trigger_bw', torch.tensor([0.0], requires_grad=False))

    def measurement_point_dim(self):
        return self.output_dim

    def generate_measuring_points_torch(self, indices):
        raise NotImplementedError()

    def forward_torch(self, *args):
        if len(args) == 0 or args[0] is None:
            dims = self.index_shape[:self.input_dim]
            sensors = torch.cartesian_prod(
                *[torch.arange(0, d, dtype=torch.long, device=torch.device('cuda:0')) for d in dims])
        else:
            sensors, = args
        return self.generate_measuring_points_torch(sensors)

    def forward(self, *args):
        '''
        Generates random measurement points for all or selected batch of sensors.
        :param args: empty set or batch of sensors.
        :return: a tensor with all points where measurement should be taken.
        '''
        if len(args) == 0 or args[0] is None:
            sensors = None
        else:
            sensors, = args
        return self.capture(self.identity, sensors, None, 1, 1)

    def capture_torch(self, field: 'MapBase', sensors_batch: Optional[torch.Tensor] = None, fw_samples: int = 1):
        # TODO: Implement a torch base replay backpropagation for stochastic processes
        output = None
        for _ in range(fw_samples):
            points = self.forward_torch(sensors_batch)
            if output is None:
                output = field.forward_torch(points)
            else:
                output += field.forward_torch(points)
        output /= fw_samples
        if sensors_batch is None:
            # reshape
            output = output.view(*self.index_shape[:self.input_dim], -1)
        return output

    def capture(self, field: 'MapBase', sensors_batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None,
                fw_samples: int = 1, bw_samples: int = 1):
        if not __USE_VULKAN_DISPATCHER__:
            return self.capture_torch(field, sensors_batch, fw_samples)
        trigger_bw = self._rdv_trigger_bw if any(True for _ in self.parameters()) or any(
            True for _ in field.parameters()) else self._rdv_no_trigger_bw
        return AutogradCaptureFunction.apply(sensors_batch, trigger_bw, batch_size, fw_samples, bw_samples, field, self)

    def random_sensors(self, batch_size: int, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        # if out is not None:
        #     out.copy_((torch.rand(batch_size, self.input_dim, device=device()) * self.index_shape[:self.input_dim].to(
        #         device())).long())
        # else:
        #     out = (torch.rand(batch_size, self.input_dim, device=device()) * self.index_shape[:self.input_dim].to(
        #         device())).long()
        # return out
        if out is None:
            # out = torch.zeros(batch_size, self.input_dim, dtype=torch.long, device=device())
            out = tensor(batch_size, self.input_dim, dtype=torch.long)
        return random_ids(batch_size, self.index_shape[:self.input_dim], out=out)


class PerspectiveCameraSensor(SensorsBase):
    __extension_info__ = dict(
        parameters=dict(
            poses=torch.Tensor,
            width=int,
            height=int,
            generation_mode=int,
            fov=float,
            znear=float
        ),
        generics=dict(OUTPUT_DIM=6),
        nodiff=True,
        path=__INCLUDE_PATH__+"/maps/camera_perspective.h"
    )

    def __init__(self, width: int, height: int, poses: torch.Tensor, jittered: bool = False):
        super(PerspectiveCameraSensor, self).__init__([len(poses), height, width])
        self.poses = poses
        self.fov = np.pi / 4
        self.znear = 0.001
        self.width = width
        self.height = height
        self.generation_mode = 0 if not jittered else 1

    def generate_measuring_points_torch(self, indices):
        o = self.poses[indices[:, 0], 0:3]
        d = self.poses[indices[:, 0], 3:6]
        n = self.poses[indices[:, 0], 6:9]
        dim = torch.tensor([self.height, self.width], dtype=torch.float, device=indices.device)
        s = ((indices[:, 1:3] + 0.5) * 2 - dim) * self.znear / self.height
        t = np.float32(self.znear / np.float32(np.tan(np.float32(self.fov) * np.float32(0.5))))
        zaxis = vec3.normalize(d)
        xaxis = vec3.normalize(vec3.cross(n, zaxis))
        yaxis = vec3.cross(zaxis, xaxis)
        w = xaxis * s[:, 1:2] + yaxis * s[:, 0:1] + zaxis * t
        x = o + w
        w = vec3.normalize(w)
        return torch.cat([x, w], dim=-1)


# # Map Operations
#
class Composition(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            inner=MapBase,
            outter=MapBase
        ),
        code="""
FORWARD {
    float intermediate [INTERMEDIATE_DIM];
    forward(parameters.inner, _input, intermediate);
    forward(parameters.outter, intermediate, _output);    
}

BACKWARD {
    float intermediate [INTERMEDIATE_DIM];
    forward(parameters.inner, _input, intermediate);
    float intermediate_grad [INTERMEDIATE_DIM]; [[unroll]] for (int i=0; i<INPUT_DIM; i++) intermediate_grad[i] = 0.0;
    backward(parameters.outter, intermediate, _output_grad, intermediate_grad);
    backward(parameters.inner, _input, intermediate_grad, _input_grad);
}
""",
    )

    def __init__(self, inner: MapBase, outter: MapBase):
        super(Composition, self).__init__(INPUT_DIM=inner.input_dim, INTERMEDIATE_DIM=inner.output_dim,
                                          OUTPUT_DIM=outter.output_dim)
        self.inner = inner
        self.outter = outter

    def forward_torch(self, *args):
        return self.outter.forward_torch(self.inner.forward_torch(*args))


class BinaryOpMap(MapBase):
    __extension_info__ = None  # Mark as an abstract map

    @staticmethod
    def create_extension_info(operation_code, requires_input, backward_code):
        return dict(
            parameters=dict(
                map_a=MapBase,
                map_b=MapBase
            ),
            code=f"""
FORWARD {{
    float _temp[OUTPUT_DIM];
    forward(parameters.map_a, _input, _output);
    forward(parameters.map_b, _input, _temp);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] {operation_code}= _temp[i];  
}}       
BACKWARD {{
    {('float a[OUTPUT_DIM]; forward(parameters.map_a, _input, a); float b[OUTPUT_DIM]; forward(parameters.map_b, _input, b);') if requires_input else ''}
    {backward_code}
}}
"""
        )

    def __init__(self, map_a: MapBase, map_b: MapBase):
        super(BinaryOpMap, self).__init__(INPUT_DIM=map_a.input_dim, OUTPUT_DIM=map_a.output_dim)
        self.map_a = map_a
        self.map_b = map_b

    def torch_operation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        pass

    def forward_torch(self, *args):
        return self.torch_operation(self.map_a(*args), self.map_b(*args))


class AdditionMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info('+', False, backward_code="""
backward(parameters.map_a, _input, _output_grad, _input_grad);
backward(parameters.map_b, _input, _output_grad, _input_grad);
     """)

    def __init__(self, map_a: MapBase, map_b: MapBase):
        super(AdditionMap, self).__init__(map_a, map_b)

    def torch_operation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class SubtractionMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info('-', False, backward_code="""
backward(parameters.map_a, _input, _output_grad, _input_grad);
[[unroll]] for (int i=0; i<OUTPUT_DIM; i++)
    _output_grad[i] *= -1;
backward(parameters.map_b, _input, _output_grad, _input_grad);
     """)

    def __init__(self, map_a: MapBase, map_b: MapBase):
        super(SubtractionMap, self).__init__(map_a, map_b)

    def torch_operation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - b


class MultiplicationMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info('*', True, backward_code="""
float dL_darg[OUTPUT_DIM];
[[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] * b[i];
backward(parameters.map_a, _input, dL_darg, _input_grad);
[[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] * a[i];
backward(parameters.map_b, _input, dL_darg, _input_grad);
    """)

    def __init__(self, map_a: MapBase, map_b: MapBase):
        super(MultiplicationMap, self).__init__(map_a, map_b)

    def torch_operation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class DivisionMap(BinaryOpMap):
    __extension_info__ = BinaryOpMap.create_extension_info('/', True, backward_code="""
float dL_darg[OUTPUT_DIM];
[[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] / b[i];
backward(parameters.map_a, _input, dL_darg, _input_grad);
[[unroll]] for (int i=0; i<OUTPUT_DIM; i++) b[i] *= b[i];
[[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] * a[i] / b[i]; // already squared
backward(parameters.map_b, _input, dL_darg, _input_grad);
    """)

    def __init__(self, map_a: MapBase, map_b: MapBase):
        super(DivisionMap, self).__init__(map_a, map_b)

    def torch_operation(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a / b


class PromoteMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(map=MapBase),
        code="""
FORWARD {
    float r[1];
    forward(parameters.map, _input, r);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = r[0];
}
BACKWARD {
    float dL_dr[1];
    dL_dr[0] = 0.0;
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        dL_dr[0] += _output_grad[i];
    backward(parameters.map, _input, dL_dr, _input_grad);
}
"""
    )

    def __init__(self, map: MapBase, dim: int):
        super(PromoteMap, self).__init__(INPUT_DIM=map.input_dim, OUTPUT_DIM=dim)
        object.__setattr__(self, 'dim', dim)
        self.map = map
        assert map.output_dim == 1, 'Promotion is only valid for single valued maps'

    def forward_torch(self, *args):
        t = self.map.forward_torch(*args)
        return t.repeat([1] * len(t.shape[:-1]), self.dim)


class ConstantMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            value=ParameterDescriptor
        ),
        code="""
FORWARD {
    float_ptr data_ptr = float_ptr(parameters.value.data);
    [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) _output[i] = data_ptr.data[i];
}
BACKWARD {
    if (parameters.value.grad_data == 0) return;
    float_ptr grad_data = float_ptr(parameters.value.grad_data);
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++)
        atomicAdd_f(grad_data, i, _output_grad[i]);
}
"""
    )

    def __init__(self, input_dim: int, value: Union[torch.Tensor, torch.nn.Parameter]):
        assert len(value.shape) == 1
        super(ConstantMap, self).__init__(INPUT_DIM=input_dim, OUTPUT_DIM=value.numel())
        self.value = parameter(value)

    def forward_torch(self, *args):
        t, = args
        return self.value.repeat(*t.shape[:-1], 1)


class ConcatMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            map_a=MapBase,
            map_b=MapBase
        ),
        code="""
    FORWARD {
        float output_a[A_OUTPUT_DIM];
        forward(parameters.map_a, _input, output_a);
        float output_b[B_OUTPUT_DIM];
        forward(parameters.map_b, _input, output_b);
        [[unroll]] for (int i=0; i < A_OUTPUT_DIM; i++) _output[i] = output_a[i];
        [[unroll]] for (int i=0; i < B_OUTPUT_DIM; i++) _output[i + A_OUTPUT_DIM] = output_b[i];
    }
    BACKWARD {
        float output_a_grad[A_OUTPUT_DIM];
        [[unroll]] for (int i=0; i < A_OUTPUT_DIM; i++) output_a_grad[i] = _output_grad[i];
        backward(parameters.map_a, _input, output_a_grad, _input_grad);
        float output_b_grad[B_OUTPUT_DIM];
        [[unroll]] for (int i=0; i < B_OUTPUT_DIM; i++) output_b_grad[i] = _output_grad[i + A_OUTPUT_DIM];
        backward(parameters.map_b, _input, output_b_grad, _input_grad);
    }
    """
    )

    def __init__(self, map_a: MapBase, map_b: MapBase):
        assert map_a.input_dim == map_b.input_dim
        super(ConcatMap, self).__init__(
            INPUT_DIM=map_a.input_dim,
            OUTPUT_DIM=map_a.output_dim + map_b.output_dim,
            A_OUTPUT_DIM=map_a.output_dim,
            B_OUTPUT_DIM=map_b.output_dim
        )
        self.map_a = map_a
        self.map_b = map_b


class IndexMap(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            map=MapBase,
            indices=[32, int]
        ),
        code="""
        FORWARD {
            float output_map[MAP_OUTPUT_DIM];
            forward(parameters.map, _input, output_map);
            [[unroll]] for (int i=0; i < MAP_OUTPUT_DIM; i++) _output[i] = output_map[parameters.indices[i]];
        }
        BACKWARD {
            float output_map_grad[MAP_OUTPUT_DIM];
            [[unroll]] for (int i=0; i < MAP_OUTPUT_DIM; i++) output_map_grad[i] = 0.0;
            [[unroll]] for (int i=0; i < OUTPUT_DIM; i++) output_map_grad[parameters.indices[i]] += _output_grad[i];
            backward(parameters.map, _input, output_map_grad, _input_grad);
        }
        """
    )

    def __init__(self, map: MapBase, indices: List[int]):
        assert all(i<map.output_dim for i in indices)
        super(IndexMap, self).__init__(
            INPUT_DIM=map.input_dim,
            OUTPUT_DIM=len(indices),
            MAP_OUTPUT_DIM=map.output_dim
        )
        self.map = map
        for i in range(len(indices)):
            self.indices[i] = indices[i]


# Sensors

class CameraSensor(SensorsBase):
    __extension_info__ = dict(
        parameters=dict(
            origin=torch.Tensor,
            direction=torch.Tensor,
            normal=torch.Tensor,
            width=int,
            height=int,
            generation_mode=int,
            fov=float,
            znear=float
        ),
        generics=dict(OUTPUT_DIM=6),
        nodiff=True,
        code=f"""
FORWARD
{{
    ivec3 index = floatBitsToInt(vec3(_input[0], _input[1], _input[2]));
    vec3_ptr origin_buf = vec3_ptr(parameters.origin);
    vec3_ptr direction_buf = vec3_ptr(parameters.direction);
    vec3_ptr normal_buf = vec3_ptr(parameters.normal);
    vec3 o = origin_buf.data[index[0]];
    vec3 d = direction_buf.data[index[0]];
    vec3 n = normal_buf.data[index[0]];

    vec2 subsample = vec2(0.5);
    if (parameters.generation_mode == 1)
        subsample = vec2(random(), random());

    float sx = ((index[2] + subsample.x) * 2 - parameters.width) * parameters.znear / parameters.height;
    float sy = ((index[1] + subsample.y) * 2 - parameters.height) * parameters.znear / parameters.height;
    float sz = parameters.znear / tan(parameters.fov * 0.5);

    vec3 zaxis = normalize(d);
    vec3 xaxis = normalize(cross(n, zaxis));
    vec3 yaxis = cross(zaxis, xaxis);

    vec3 x, w;

    w = xaxis * sx + yaxis * sy + zaxis * sz;
    x = o + w;
    w = normalize(w);

    _output = float[6]( x.x, x.y, x.z, w.x, w.y, w.z );
}}
"""
    )

    def __init__(self, width: int, height: int, cameras: int = 1, jittered: bool = False):
        super(CameraSensor, self).__init__([cameras, height, width])
        self.origin = tensor(cameras, 3, dtype=torch.float32)
        self.direction = tensor(cameras, 3, dtype=torch.float32)
        self.normal = tensor(cameras, 3, dtype=torch.float32)
        self.origin[:] = 0.0
        self.origin[:, 2] = -1.0
        self.direction[:] = 0.0
        self.direction[:, 2] = 1.0
        self.normal[:] = 0.0
        self.normal[:, 1] = 1.0
        self.fov = np.pi / 4
        self.znear = 0.001
        self.width = width
        self.height = height
        self.generation_mode = 0 if not jittered else 1

    def generate_measuring_points_torch(self, indices):
        o = self.origin[indices[:, 0]]
        d = self.direction[indices[:, 0]]
        n = self.normal[indices[:, 0]]
        dim = torch.tensor([self.height, self.width], dtype=torch.float, device=indices.device)
        s = ((indices[:, 1:3] + 0.5) * 2 - dim) * self.znear / self.height
        t = np.float32(self.znear / np.float32(np.tan(np.float32(self.fov) * np.float32(0.5))))
        zaxis = vec3.normalize(d)
        xaxis = vec3.normalize(vec3.cross(n, zaxis))
        yaxis = vec3.cross(zaxis, xaxis)
        w = xaxis * s[:, 1:2] + yaxis * s[:, 0:1] + zaxis * t
        x = o + w
        w = vec3.normalize(w)
        return torch.cat([x, w], dim=-1)


class Grid3DSensor(SensorsBase):
    __extension_info__ = dict(
        parameters=dict(
            width=int,
            height=int,
            depth=int,
            box_min=vec3,
            box_max=vec3,
            sd=float
        ),
        generics=dict(OUTPUT_DIM=3),
        nodiff=True,
        code=f"""
FORWARD
{{
    ivec3 index = floatBitsToInt(vec3(_input[0], _input[1], _input[2]));
    vec3 subsample = vec3(0.0);
    if (parameters.sd != 0)
        subsample = gauss3() * parameters.sd;

    vec3 sx = vec3((index[2] + subsample.x) / (parameters.width - 1), (index[1] + subsample.y) / (parameters.height - 1), (index[0] + subsample.z) / (parameters.depth - 1));

    sx = (parameters.box_max - parameters.box_min) * sx + parameters.box_min;        
    _output = float[3]( sx.x, sx.y, sx.z );
}}
    """
    )

    def __init__(self, width: int, height: int, depth: int = 1, box_min: vec3 = vec3(-1.0, -1.0, -1.0),
                 box_max: vec3 = vec3(1.0, 1.0, 1.0), sd: float = 0.0):
        super(Grid3DSensor, self).__init__([depth, height, width])
        self.width = width
        self.height = height
        self.depth = depth
        self.sd = sd
        self.box_min = box_min.clone()
        self.box_max = box_max.clone()

    def generate_measuring_points_torch(self, indices):
        dim = torch.tensor([self.depth - 1, self.height - 1, self.width - 1], dtype=torch.float, device=indices.device)
        subsample = torch.randn(*indices.shape, device=indices.device) * self.sd
        bmin = self.box_min.to(indices.device)
        bmax = self.box_max.to(indices.device)
        return ((indices + subsample) / dim) * (bmax - bmin) + bmin


class Box3DSensor(SensorsBase):
    __extension_info__ = dict(
        parameters=dict(
            count=int,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(OUTPUT_DIM=3),
        nodiff=True,
        code=f"""
FORWARD
{{
    vec3 sx = (parameters.box_max - parameters.box_min) * vec3(random(), random(), random()) + parameters.box_min;        
    _output = float[3]( sx.x, sx.y, sx.z );
}}
    """
    )

    def __init__(self, samples, box_min: vec3 = vec3(-1.0, -1.0, -1.0), box_max: vec3 = vec3(1.0, 1.0, 1.0)):
        super(Box3DSensor, self).__init__([samples])
        self.box_min = box_min.clone()
        self.box_max = box_max.clone()


class DummyExample(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            alpha=float
        ),
        code="""
FORWARD {
    [[unroll]]
    for (int i=0; i<INPUT_DIM; i++)
        _output[i] = (_input[i] * parameters.alpha + _input[i]) * parameters.alpha;
}               
        """,
        nodiff=True,
        use_raycast=False,
    )
    
    def __init__(self, dimension: int, alpha: float = 1.0):
        super(DummyExample, self).__init__(INPUT_DIM=dimension, OUTPUT_DIM=dimension)
        # parameter fields requires a 'proxy' attribute in the module
        self.alpha = alpha

    def forward_torch(self, *args):
        return (args[0] * self.alpha + args[0])*self.alpha


class Grid2D(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            bmin=vec2,
            inv_bsize=vec2
        ),
        generics=dict(INPUT_DIM=2),
        nodiff=True,
        code="""
FORWARD
{
    vec2 c = vec2(_input[0], _input[1]);
    vec2 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    if (any(lessThan(ncoord, vec2(0.0))) || any(greaterThanEqual(ncoord, vec2(1.0)))) {
        [[unroll]]
        for (int i=0; i<OUTPUT_DIM; i++)
            _output[i] = 0.0;
        return;
    }
    vec2 grid_coord = ncoord * vec2(parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    ivec2 p = ivec2(floor(grid_coord));
    vec2 alpha = grid_coord - p;
    float_ptr grid_buf_00 = param_buffer(parameters.grid, p + ivec2(0,0)); 
    float_ptr grid_buf_01 = param_buffer(parameters.grid, p + ivec2(1,0)); 
    float_ptr grid_buf_10 = param_buffer(parameters.grid, p + ivec2(0,1)); 
    float_ptr grid_buf_11 = param_buffer(parameters.grid, p + ivec2(1,1)); 
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = mix(
            mix(grid_buf_00.data[i], grid_buf_01.data[i], alpha.x),
            mix(grid_buf_10.data[i], grid_buf_11.data[i], alpha.x), alpha.y
        );
}

"""
    )

    def __init__(self, grid: Union[torch.Tensor, torch.nn.Parameter], bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
        assert len(grid.shape) == 3
        super(Grid2D, self).__init__(OUTPUT_DIM=grid.shape[-1])
        self.grid = parameter(grid)
        self.bmin = bmin.clone()
        self.inv_bsize = (1.0/(bmax- bmin)).clone()

    def forward_torch(self, *args):
        x, = args
        x = 2 * (x - self.bmin.to(self.grid.device)) * self.inv_bsize.to(self.grid.device) - 1
        return torch.nn.functional.grid_sample(
            self.grid.unsqueeze(0).permute(0, 3, 1, 2),
            x.reshape(1, len(x), 1, -1),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(-1, self.output_dim)


class Image2D(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            bmin=vec2,
            inv_bsize=vec2
        ),
        generics=dict(INPUT_DIM=2),
        nodiff=True,
        code="""
FORWARD
{
    vec2 c = vec2(_input[0], _input[1]);
    vec2 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    if (any(lessThan(ncoord, vec2(0.0))) || any(greaterThanEqual(ncoord, vec2(1.0)))) {
        [[unroll]]
        for (int i=0; i<OUTPUT_DIM; i++)
            _output[i] = 0.0;
        return;
    }
    vec2 grid_coord = (ncoord * vec2(parameters.grid.shape[1], parameters.grid.shape[0]) - vec2(0.5));
    ivec2 max_dim = ivec2(parameters.grid.shape[1], parameters.grid.shape[0]) - 1;
    ivec2 p = ivec2(floor(grid_coord));
    vec2 alpha = grid_coord - p;
    float_ptr grid_buf_00 = param_buffer(parameters.grid, clamp(p + ivec2(0,0), ivec2(0), max_dim)); 
    float_ptr grid_buf_01 = param_buffer(parameters.grid, clamp(p + ivec2(1,0), ivec2(0), max_dim)); 
    float_ptr grid_buf_10 = param_buffer(parameters.grid, clamp(p + ivec2(0,1), ivec2(0), max_dim)); 
    float_ptr grid_buf_11 = param_buffer(parameters.grid, clamp(p + ivec2(1,1), ivec2(0), max_dim)); 
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = mix(
            mix(grid_buf_00.data[i], grid_buf_01.data[i], alpha.x),
            mix(grid_buf_10.data[i], grid_buf_11.data[i], alpha.x), alpha.y
        );
}

"""
    )

    def __init__(self, grid: Union[torch.Tensor, torch.nn.Parameter], bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
        assert len(grid.shape) == 3
        super(Image2D, self).__init__(OUTPUT_DIM=grid.shape[-1])
        self.grid = parameter(grid)
        self.bmin = bmin.clone()
        self.inv_bsize = (1.0/(bmax- bmin)).clone()

    def forward_torch(self, *args):
        x, = args
        x = 2 * (x - self.bmin.to(self.grid.device)) * self.inv_bsize.to(self.grid.device) - 1
        return torch.nn.functional.grid_sample(
            self.grid.unsqueeze(0).permute(0, 3, 1, 2),
            x.reshape(1, len(x), 1, -1),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(-1, self.output_dim)


class Grid3D(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=3),
        parameters=dict(
            grid=ParameterDescriptor,
            bmin=vec3,
            inv_bsize=vec3
        ),
        code="""
        
void blend(map_object, inout float dst[OUTPUT_DIM], float_ptr src, float alpha)
{
    [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dst[i] += src.data[i] * alpha;
} 

FORWARD
{
    vec3 c = vec3(_input[0], _input[1], _input[2]);
    vec3 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = 0.0;
    if (any(lessThan(ncoord, vec3(0.0))) || any(greaterThanEqual(ncoord, vec3(1.0))))
    {
        return;
    }
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 grid_coord = ncoord * vec3(dim);
    vec3 alpha = fract(grid_coord);
    ivec3 p = clamp(ivec3(grid_coord), ivec3(0), dim - 1);
    float_ptr g = param_buffer(parameters.grid, p + ivec3(0,0,0));
    blend(object, _output, g, (1 - alpha.x)*(1 - alpha.y)*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(1,0,0));
    blend(object, _output, g, alpha.x*(1 - alpha.y)*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(0,1,0));
    blend(object, _output, g, (1 - alpha.x)*alpha.y*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(1,1,0));
    blend(object, _output, g, alpha.x*alpha.y*(1 - alpha.z));
    g = param_buffer(parameters.grid, p + ivec3(0,0,1));
    blend(object, _output, g, (1 - alpha.x)*(1 - alpha.y)*alpha.z);
    g = param_buffer(parameters.grid, p + ivec3(1,0,1));
    blend(object, _output, g, alpha.x*(1 - alpha.y)*alpha.z);
    g = param_buffer(parameters.grid, p + ivec3(0,1,1));
    blend(object, _output, g, (1 - alpha.x)*alpha.y*alpha.z);
    g = param_buffer(parameters.grid, p + ivec3(1,1,1));
    blend(object, _output, g, alpha.x*alpha.y*alpha.z);
}

BACKWARD
{
    if (parameters.grid.grad_data == 0) // NULL GRAD
        return;
    
    vec3 c = vec3(_input[0], _input[1], _input[2]);
    vec3 ncoord = (c - parameters.bmin) * parameters.inv_bsize;
    
    if (any(lessThan(ncoord, vec3(0.0))) || any(greaterThanEqual(ncoord, vec3(1.0))))
        return;

    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 grid_coord = ncoord * vec3(dim);
    vec3 alpha = fract(grid_coord);
    ivec3 p = clamp(ivec3(grid_coord), ivec3(0), dim - 1);
    float_ptr g000 = param_grad_buffer(parameters.grid, p + ivec3(0,0,0));
    float_ptr g001 = param_grad_buffer(parameters.grid, p + ivec3(1,0,0));
    float_ptr g010 = param_grad_buffer(parameters.grid, p + ivec3(0,1,0));
    float_ptr g011 = param_grad_buffer(parameters.grid, p + ivec3(1,1,0));
    float_ptr g100 = param_grad_buffer(parameters.grid, p + ivec3(0,0,1));
    float_ptr g101 = param_grad_buffer(parameters.grid, p + ivec3(1,0,1));
    float_ptr g110 = param_grad_buffer(parameters.grid, p + ivec3(0,1,1));
    float_ptr g111 = param_grad_buffer(parameters.grid, p + ivec3(1,1,1));
    
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
    {
        atomicAdd_f(g000, i, _output_grad[i] * (1 - alpha.x) * (1 - alpha.y) * (1 - alpha.z));
        atomicAdd_f(g001, i, _output_grad[i] * (alpha.x) * (1 - alpha.y) * (1 - alpha.z));
        atomicAdd_f(g010, i, _output_grad[i] * (1 - alpha.x) * (alpha.y) * (1 - alpha.z));
        atomicAdd_f(g011, i, _output_grad[i] * (alpha.x) * (alpha.y) * (1 - alpha.z));
        atomicAdd_f(g100, i, _output_grad[i] * (1 - alpha.x) * (1 - alpha.y) * (alpha.z));
        atomicAdd_f(g101, i, _output_grad[i] * (alpha.x) * (1 - alpha.y) * (alpha.z));
        atomicAdd_f(g110, i, _output_grad[i] * (1 - alpha.x) * (alpha.y) * (alpha.z));
        atomicAdd_f(g111, i, _output_grad[i] * (alpha.x) * (alpha.y) * (alpha.z));
    }
}
"""
    )

    def __init__(self, grid: torch.Tensor, bmin: vec3 = vec3(-1.0, -1.0, -1.0), bmax: vec3 = vec3(1.0, 1.0, 1.0)):
        assert len(grid.shape) == 4
        super(Grid3D, self).__init__(OUTPUT_DIM=grid.shape[-1])
        self.grid = parameter(grid)
        self.bmin = bmin.clone()
        self.inv_bsize = (1.0/(bmax- bmin)).clone()
        self.bmax = bmax.clone()
        self.width = grid.shape[2]
        self.height = grid.shape[1]
        self.depth = grid.shape[0]

    def forward_torch(self, *args):
        x, = args
        x = 2 * (x - self.bmin.to(self.grid.device)) * self.inv_bsize.to(self.grid.device) - 1
        return torch.nn.functional.grid_sample(
            self.grid.unsqueeze(0).permute(0, 4, 1, 2, 3),
            x.reshape(1, len(x), 1, 1, -1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).permute(0, 2, 3, 4, 1).view(-1, self.output_dim)

    def line_integral(self):
        # return LineIntegrator(self)
        # return old_Grid3DLineIntegral(self)
        return Grid3DLineIntegral(self)

    def to_transmittance(self):
        return TransmittanceFromTau(self.line_integral())

    def get_maximum(self) -> float:
        return self.grid.max().item()

    def support_maximum_query(self) -> bool:
        return True


class XRProjection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            OUTPUT_DIM=2
        ),
        code = """
FORWARD
{
#if INPUT_DIM == 3
    _input[0] += 0.0000001 * int(_input[2] == 0.0 && _input[0] == 0.0);
    _output[0] = atan(_input[0], _input[2]) * inverseOfPi;
    _output[1] = 2 * acos(clamp(_input[1], -1.0, 1.0)) / pi - 1;
#else
    _input[3] += 0.0000001 * int(_input[5] == 0.0 && _input[3] == 0.0);
    _output[0] = atan(_input[3], _input[5]) * inverseOfPi;
    _output[1] = 2 * acos(clamp(_input[4], -1.0, 1.0)) / pi - 1;
#endif
}
""",
        nodiff=True,
    )

    def __init__(self, ray_input: bool = False):
        super(XRProjection, self).__init__(INPUT_DIM=6 if ray_input else 3)
        object.__setattr__(self, 'input_ray', 1 if ray_input else 0)

    def forward_torch(self, *args):
        if self.input_ray:
            xw, = args
            w = xw[:, 3:6]
        else:
            w, = args
        #    vec2 c = vec2((atan(w.z, w.x) + pi) / (2 * pi), acos(clamp(w.y, -1.0, 1.0)) / pi); // two floats for coordinates
        a = (torch.atan2(w[:,0:1], w[:,2:3])) / np.pi
        b = 2 * torch.acos(torch.clamp(w[:, 1:2], -1.0, 1.0)) / np.pi - 1
        return torch.cat([a, b], dim=-1)


class UniformRandom(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        code="""
        FORWARD
        {
            [[unroll]] for (int i=0; i<OUTPUT_DIM - 1; i++) _output[i] = random();
            _output[OUTPUT_DIM-1] = 1.0;
        }
                """, nodiff=True
    )

    def __init__(self, input_dim, point_dim):
        super(UniformRandom, self).__init__(INPUT_DIM=input_dim, OUTPUT_DIM=point_dim + 1)


class GaussianRandom(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        code="""
FORWARD
{
    float sum = 0; 
    [[unroll]] for (int i=0; i<(OUTPUT_DIM - 1) / 2; i++)
    {
        vec2 r = gauss2();
        _output[i*2] = r.x;
        _output[i*2 + 1] = r.y;
        sum += dot(r, r);
    }
    if (OUTPUT_DIM % 2 == 0)
    {
        float r = gauss();
        _output[OUTPUT_DIM-2] = r;
        sum += r * r;
    }
    _output[OUTPUT_DIM-1] = pow(two_pi, 0.5 * (OUTPUT_DIM-1)) * exp (0.5 * sum); // 1 / pdf(x)
}
                    """, nodiff=True
    )

    def __init__(self, input_dim, output_dim):
        super(GaussianRandom, self).__init__(INPUT_DIM=input_dim, OUTPUT_DIM=output_dim)


class UniformRandomDirection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(OUTPUT_DIM=4),
        code="""
    FORWARD
    {
        vec3 w_out = randomDirection();
        _output = float[4](w_out.x, w_out.y, w_out.z, 4 * pi);
    }
            """, nodiff=True
    )

    def __init__(self, input_dim):
        super(UniformRandomDirection, self).__init__(INPUT_DIM=input_dim)


class XRQuadtreeRandomDirection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            densities=torch.Tensor,
            levels=int
        ),
        generics=dict(OUTPUT_DIM=4),
        code="""
    FORWARD
    {
        float_ptr densities_buf = float_ptr(parameters.densities);
        float sel = random();
        int current_node = 0;
        vec2 p0 = vec2(0,0);
        vec2 p1 = vec2(1,1);
        float prob = 1;
        [[unroll]]
        for (int i=0; i<parameters.levels; i++)
        {
            int offset = current_node * 4;
            int selected_child = 3;
            prob = densities_buf.data[offset + 3];
            [[unroll]]
            for (int c = 0; c < 3; c ++)
                if (sel < densities_buf.data[offset + c])
                {
                    selected_child = c;
                    prob = densities_buf.data[offset + c];
                    break;
                }
                else
                    sel -= densities_buf.data[offset + c];;
            float xmed = (p1.x + p0.x)/2;
            float ymed = (p1.y + p0.y)/2;
            if (selected_child % 2 == 0) // left
                p1.x = xmed;
            else
                p0.x = xmed;
            if (selected_child / 2 == 0) // top
                p1.y = ymed;
            else
                p0.y = ymed;
            current_node = current_node * 4 + 1 + selected_child;
        }
        float pixel_area = 2 * pi * (cos(p0.y*pi) - cos(p1.y*pi)) / (1 << parameters.levels);
        float weight = pixel_area / max(0.000000001, prob);
        vec3 w_out = randomDirection((p0.x * 2 - 1) * pi, (p1.x * 2 - 1) * pi, p0.y * pi, p1.y * pi);
        _output = float[4](w_out.x, w_out.y, w_out.z, weight);
    }
            """, nodiff=True
    )

    def __init__(self, input_dim: int, densities: torch.Tensor, levels: int):
        super(XRQuadtreeRandomDirection, self).__init__(INPUT_DIM=input_dim)
        self.densities = densities
        self.levels = levels


class FunctionSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            point_sampler=MapBase,
            function_map=MapBase
        ),
        generics=dict(),
        code = """
FORWARD
{
    float x_wx [FUNCTION_INPUT_DIM + 1]; // x: FUNCTION_INPUT_DIM, wx: 1
    forward(parameters.point_sampler, _input, x_wx);
    
    float function_in[FUNCTION_INPUT_DIM];
    [[unroll]]
    for (int i=0; i<FUNCTION_INPUT_DIM; i++) function_in[i] = x_wx[i];
    float wx = x_wx[FUNCTION_INPUT_DIM];
    float function_out[FUNCTION_OUTPUT_DIM];
    forward(parameters.function_map, function_in, function_out);
    [[unroll]] for (int i = 0; i<FUNCTION_OUTPUT_DIM; i++) _output[i] = function_out[i] * wx;
    [[unroll]] for (int i = 0; i<FUNCTION_INPUT_DIM; i++) _output[i + FUNCTION_OUTPUT_DIM] = function_in[i];
}
        """, nodiff=True
    )
    def __init__(self, point_sampler: MapBase, function_map: MapBase):
        assert point_sampler.output_dim - 1 == function_map.input_dim
        super(FunctionSampler, self).__init__(
            INPUT_DIM=point_sampler.input_dim,
            OUTPUT_DIM=point_sampler.output_dim - 1 + function_map.output_dim,
            FUNCTION_INPUT_DIM=function_map.input_dim,
            FUNCTION_OUTPUT_DIM=function_map.output_dim
        )
        self.point_sampler = point_sampler
        self.function_map = function_map


class GridRatiotrackingTransmittance(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3,
            boundary=MapBase
        ),
        path=__INCLUDE_PATH__ + "/maps/transmittance_grt.h",
        nodiff=True
    )

    def __init__(self, grid: Grid3D, boundary: MapBase):
        super(GridRatiotrackingTransmittance, self).__init__()
        self.grid_model = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()
        self.boundary = boundary


class GridDeltatrackingTransmittance(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3,
            boundary=MapBase
        ),
        path=__INCLUDE_PATH__ + "/maps/transmittance_gdt.h",
        nodiff=True
    )

    def __init__(self, grid: Grid3D, boundary: MapBase):
        super(GridDeltatrackingTransmittance, self).__init__()
        self.grid_model = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()
        self.boundary = boundary


class GridDDATransmittance(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3,
            boundary=MapBase
        ),
        path=__INCLUDE_PATH__ + "/maps/transmittance_dda.h",
        # nodiff=True
    )

    def __init__(self, grid: Grid3D, boundary: MapBase):
        super(GridDDATransmittance, self).__init__()
        self.grid_model = grid
        self.grid = grid.grid
        self.box_min = grid.bmin.clone()
        self.box_max = grid.bmax.clone()
        self.boundary = boundary


class RatiotrackingTransmittance(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + "/maps/transmittance_rt.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(sigma=MapBase, boundary=MapBase, majorant=MapBase),
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, majorant: MapBase):
        super(RatiotrackingTransmittance, self).__init__()
        self.sigma = sigma
        self.boundary = boundary
        self.majorant = majorant


class DeltatrackingTransmittance(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + "/maps/transmittance_dt.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(sigma=MapBase, boundary=MapBase, majorant=MapBase),
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, majorant: MapBase):
        super(DeltatrackingTransmittance, self).__init__()
        self.sigma = sigma
        self.boundary = boundary
        self.majorant = majorant


class RaymarchingTransmittance(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + "/maps/transmittance_rm.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        parameters=dict(sigma=MapBase, boundary=MapBase, step=float),
        nodiff=True
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, step: float = 0.005):
        super(RaymarchingTransmittance, self).__init__()
        self.sigma = sigma
        self.boundary = boundary
        self.step = step


class DeltatrackingCollisionSampler(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + "/maps/collision_sampler_dt.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(sigma=MapBase, boundary=MapBase, majorant=MapBase, ds_epsilon=float),
    )

    def __init__(self, sigma: MapBase, boundary: MapBase, majorant: MapBase, ds_epsilon:float):
        super(DeltatrackingCollisionSampler, self).__init__()
        self.sigma = sigma
        self.boundary = boundary
        self.majorant = majorant
        self.ds_epsilon = ds_epsilon


class MCScatteredRadiance(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + "/maps/scattered_radiance_mc.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(scattering_albedo=MapBase, phase_sampler=MapBase, radiance=MapBase),
    )

    def __init__(self, scattering_albedo: MapBase, phase_sampler: MapBase, radiance: MapBase):
        super(MCScatteredRadiance, self).__init__()
        self.scattering_albedo = scattering_albedo
        self.phase_sampler = phase_sampler
        self.radiance = radiance


class MCScatteredEmittedRadiance(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + "/maps/scattered_emitted_radiance_mc.h",
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(scattering_albedo=MapBase, emission=MapBase, phase_sampler=MapBase, radiance=MapBase),
    )

    def __init__(self, scattering_albedo: MapBase, emission: MapBase, phase_sampler: MapBase, radiance: MapBase):
        super(MCScatteredEmittedRadiance, self).__init__()
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.phase_sampler = phase_sampler
        self.radiance = radiance


class MCCollisionIntegrator(MapBase):
    '''
    Montecarlo solver for
    :math:`\int^{x_{d}}_{x_{0}} T(x, x_{t})\sigma(x_{t}) R(x_{t} , -w) dt + T(x, x_d)B(w)`
    using \n
    collision_sampler: :math:`<T(x, x_{t})\sigma(x_{t})>/p(t), t, <T(x, x_d)>` \n
    exitance_radiance: :math:`R(x, w)` \n
    environment: :math:`B(w)`
    '''
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/collision_integrator_mc.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(collision_sampler=MapBase, exitance_radiance=MapBase, environment=MapBase),
    )

    def __init__(self, collision_sampler: MapBase, exitance_radiance: MapBase, environment: MapBase):
        super(MCCollisionIntegrator, self).__init__()
        self.collision_sampler = collision_sampler
        self.exitance_radiance = exitance_radiance
        self.environment = environment


class GridDDACollisionIntegrator(MapBase):
    '''
    DDA solver for
    :math:`\int^{x_{d}}_{x_{0}} T(x, x_{t})\sigma(x_{t}) R(x_{t} , -w) dt + T(x, x_d)B(w)`
    using \n
    collision_sampler: :math:`<T(x, x_{t})\sigma(x_{t})>/p(t), t, <T(x, x_d)>` \n
    exitance_radiance: :math:`R(x, w)` \n
    environment: :math:`B(w)`
    '''
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/collision_integrator_dda.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            grid=ParameterDescriptor,
            exitance_radiance=MapBase,
            environment=MapBase,
            boundary=MapBase,
            box_min=vec3,
            box_max=vec3
        ),
    )

    def __init__(self, sigma_grid: Grid3D, exitance_radiance: MapBase, environment: MapBase, boundary: MapBase):
        super(GridDDACollisionIntegrator, self).__init__()
        self.sigma_grid = sigma_grid
        self.grid = sigma_grid.grid
        self.box_min = sigma_grid.bmin.clone()
        self.box_max = sigma_grid.bmax.clone()
        self.exitance_radiance = exitance_radiance
        self.environment = environment
        self.boundary = boundary


class DeltatrackingPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/radiance_DT.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            ds_epsilon=float
        ),
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 ds_epsilon: float
                 ):
        super(DeltatrackingPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.ds_epsilon = ds_epsilon


class DeltatrackingNEEPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/radiance_NEE_DS.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase,
            ds_epsilon=float
        ),
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase,
                 ds_epsilon: float
                 ):
        super(DeltatrackingNEEPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance
        self.ds_epsilon = ds_epsilon


class DRTPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/radiance_NEE_DRT.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase
        ),
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase
                 ):
        super(DRTPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance


class DRTQPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/radiance_NEE_DRTQ.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase
        ),
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase
                 ):
        super(DRTQPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance



class DRTDSPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/radiance_NEE_DRTDS.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase,
            ds_epsilon=float
        ),
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase,
                 ds_epsilon: float
                 ):
        super(DRTDSPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance
        self.ds_epsilon = ds_epsilon



class SPSPathIntegrator(MapBase):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/maps/radiance_NEE_SPS.h',
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            emission=MapBase,
            environment=MapBase,
            environment_sampler=MapBase,
            phase=MapBase,
            phase_sampler=MapBase,
            boundary=MapBase,
            majorant=MapBase,
            transmittance=MapBase
        ),
    )

    def __init__(self,
                 sigma: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 environment_sampler: MapBase,
                 phase: MapBase,
                 phase_sampler: MapBase,
                 boundary: MapBase,
                 majorant: MapBase,
                 transmittance: MapBase
                 ):
        super(SPSPathIntegrator, self).__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.phase = phase
        self.phase_sampler = phase_sampler
        self.boundary = boundary
        self.majorant = majorant
        self.transmittance = transmittance



# --------------------


class RayDirection(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=3
        ),
        code = """
FORWARD
{
    [[unroll]]
    for (int i=0; i<3; i++)
        _output[i] = _input[3 + i];
}
""",
        nodiff=True,
    )

    def __init__(self):
        super(RayDirection, self).__init__()

    def forward_torch(self, *args):
        xw, = args
        return xw[...,3:6]


class RayPosition(MapBase):
    __extension_info__ = dict(
        parameters=dict(
        ),
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=3
        ),
        code = """
FORWARD
{
    [[unroll]]
    for (int i=0; i<3; i++)
        _output[i] = _input[i];
}
""",
        nodiff=True,
    )

    def __init__(self):
        super(RayPosition, self).__init__()

    def forward_torch(self, *args):
        xw, = args
        return xw[...,0:3]


class RayToSegment(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            distance_field=MapBase
        ),
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=6
        ),
        code="""
    FORWARD
    {
        [[unroll]]
        for (int i=0; i<3; i++)
            _output[i] = _input[i]; // x0 = x
        float d[1];
        forward(parameters.distance_field, _input, d);
        [[unroll]]
        for (int i=0; i<3; i++)
            _output[i+3] = _input[i+3]*d[0] + _input[i]; // x1 = x + w*d
    }
    """,
        nodiff=True,
    )

    def __init__(self, distance_field: 'MapBase'):
        super(RayToSegment, self).__init__()
        self.distance_field = distance_field

    def forward_torch(self, *args):
        xw, = args
        x = xw[..., 0:3]
        w = xw[..., 3:6]
        d = self.distance_field(xw)
        return torch.cat([x, x + w*d], dim=-1)


class LineIntegrator(MapBase):

    __extension_info__ = dict(
        parameters=dict(
            map=MapBase,
            step=float
        ),
        code = """
FORWARD
{
    float x0[INPUT_DIM/2];
    [[unroll]]
    for (int i = 0; i < INPUT_DIM/2; i++)
        x0[i] = _input[i];
    float dx[INPUT_DIM/2];
    float d = 0.0;
    [[unroll]]
    for (int i = 0; i < INPUT_DIM/2; i++)
    {
        dx[i] = _input[INPUT_DIM/2 + i] - x0[i];
        d += dx[i] * dx[i];
    }
    d = sqrt(d);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = 0.0;
    int samples = int(d / (parameters.step + 0.00000001)) + 1;
    [[unroll]]
    for (int s = 0; s < samples; s++)
    {
        float xt[INPUT_DIM / 2];
        float alpha = random();
        [[unroll]]
        for (int i=0; i<INPUT_DIM/2; i++)
            xt[i] = dx[i] * alpha + x0[i];
        float o[OUTPUT_DIM];
        forward(parameters.map, xt, o);
        [[unroll]]
        for (int i=0; i<OUTPUT_DIM; i++)
            _output[i] += o[i];
    }
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] /= samples;
}
        """,
        nodiff=True,
    )

    def __init__(self, map: 'MapBase', step: float = 0.005):
        super(LineIntegrator, self).__init__(INPUT_DIM=map.input_dim * 2, OUTPUT_DIM=map.output_dim)
        self.map = map
        self.step = step


class TransmittanceDT(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            sigma=MapBase,
            majorant=float
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        code="""
    FORWARD
    {
        vec3 x0 = vec3(_input[0], _input[1], _input[2]);
        vec3 x1 = vec3(_input[3], _input[4], _input[5]);
        vec3 dx = x1 - x0;
        float d = length(dx);
        while(true)
        {
            float t = -log(1 - random()) / parameters.majorant;
            if (t >= d)
            {
                _output[0] = 1.0;
                return;
            }
            float sigma_value[1];
            x0 += dx * t;
            forward(parameters.sigma, float[3]( x0.x, x0.y, x0.z ), sigma_value);
            if (random() < sigma_value[0] / parameters.majorant)
                break;
            d -= t; 
        }
        _output[0] = 0.0;
    }
            """,
        nodiff=True,
    )

    def __init__(self, sigma: 'MapBase', majorant: float):
        super(TransmittanceDT, self).__init__()
        self.sigma = sigma
        self.majorant = majorant


class TransmittanceRT(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            sigma=MapBase,
            majorant=float
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        code="""
    FORWARD
    {
        vec3 x0 = vec3(_input[0], _input[1], _input[2]);
        vec3 x1 = vec3(_input[3], _input[4], _input[5]);
        vec3 dx = x1 - x0;
        float d = length(dx);
        float T = 1.0;
        while(true)
        {
            float t = -log(1 - random()) / parameters.majorant;
            if (t >= d)
                break;
            float sigma_value[1];
            x0 += dx * t;
            forward(parameters.sigma, float[3]( x0.x, x0.y, x0.z ), sigma_value);
            T *= (1 - sigma_value[0] / parameters.majorant);
            d -= t; 
        }
        _output[0] = T;
    }
            """,
        nodiff=True,
    )

    def __init__(self, sigma: 'MapBase', majorant: float):
        super(TransmittanceRT, self).__init__()
        self.sigma = sigma
        self.majorant = majorant


class TotalVariation(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            map=MapBase,
            expected_dx=float
        ),
        code="""
    // Total Variation - input: x, output: (map(x + dx * random_w) - map(x))/dx
    FORWARD
    {
        float w[INPUT_DIM];
        float dx = 0;
        [[unroll]]
        for (int i=0; i<INPUT_DIM; i++)
        {
            w[i] = gauss() * parameters.expected_dx;
            dx += w[i] * w[i];
            w[i] = w[i] + _input[i];
        }
        dx = sqrt(dx);

        dx = max (dx, 0.00000001);
        
        forward(parameters.map, _input, _output); // eval map at current pos
        float adj_output[OUTPUT_DIM];
        forward(parameters.map, w, adj_output);
        [[unroll]]
        for (int i = 0; i < OUTPUT_DIM; i++)
            _output[i] = abs(adj_output[i] - _output[i]) / dx;
    }
    
    BACKWARD
    {
        float w[INPUT_DIM];
        float dx = 0;
        [[unroll]]
        for (int i=0; i<INPUT_DIM; i++)
        {
            w[i] = gauss() * parameters.expected_dx;
            dx += w[i] * w[i];
            w[i] = w[i] + _input[i];
        }
        dx = sqrt(dx);

        dx = max (dx, 0.00000001);
        
        float _output[OUTPUT_DIM];
        forward(parameters.map, _input, _output); // eval map at current pos
        float adj_output[OUTPUT_DIM];
        forward(parameters.map, w, adj_output);

        float tmp_output_grad[OUTPUT_DIM];
        [[unroll]]
        for (int i = 0; i < OUTPUT_DIM; i++)
            tmp_output_grad[i] = sign(adj_output[i] - _output[i]) * _output_grad[i] / dx;
        backward(parameters.map, w, tmp_output_grad, _input_grad);
        [[unroll]]
        for (int i = 0; i < OUTPUT_DIM; i++)
            tmp_output_grad[i] = -tmp_output_grad[i];
        backward(parameters.map, _input, tmp_output_grad, _input_grad);
    }
            """
    )

    def __init__(self, map: 'MapBase', expected_dx: float = 0.005):
        super(TotalVariation, self).__init__(INPUT_DIM=map.input_dim, OUTPUT_DIM=map.output_dim)
        self.map = map
        self.expected_dx = expected_dx


class TransmittanceFromTau(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            tau=MapBase
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        code="""
    FORWARD
    {
        forward(parameters.tau, _input, _output);
        _output[0] = exp(-_output[0]);
    }
    
    BACKWARD
    {
        float dL_dT = _output_grad[0];
        float tau[1];
        forward(parameters.tau, _input, tau);
        float dL_dtau = - dL_dT * exp(-tau[0]);
        // dT/dtau = -exp(-tau) * dtau
        backward(parameters.tau, _input, float[1](dL_dtau), _input_grad);
    }
            """
    )

    def __init__(self, tau: 'MapBase'):
        super(TransmittanceFromTau, self).__init__()
        self.tau = tau


class Grid3DLineIntegral(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6),
        code="""

void load_tensor_at(map_object, ivec3 cell, out float[OUTPUT_DIM] values)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    float_ptr buf = param_buffer(parameters.grid, cell);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        values[i] = buf.data[i]; 
}

void add_grad_tensor_at(map_object, ivec3 cell, float[OUTPUT_DIM] values)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    //if (parameters.grid.grad_data == 0)
    //return;
    
    float_ptr buf = param_grad_buffer(parameters.grid, cell);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        atomicAdd_f(buf, i, values[i]);
}


void load_cell(map_object, ivec3 cell, out float[2][2][2][OUTPUT_DIM] values)
{
    [[unroll]]
    for (int dz = 0; dz < 2; dz ++)
        [[unroll]]
        for (int dy = 0; dy < 2; dy ++)
            [[unroll]]
            for (int dx = 0; dx < 2; dx ++) 
                load_tensor_at(object, cell + ivec3(dx, dy, dz), values[dz][dy][dx]);
}

void add_grad_cell(map_object, ivec3 cell, float[2][2][2][OUTPUT_DIM] values)
{
    [[unroll]]
    for (int dz = 0; dz < 2; dz ++)
        [[unroll]]
        for (int dy = 0; dy < 2; dy ++)
            [[unroll]]
            for (int dx = 0; dx < 2; dx ++) 
                add_grad_tensor_at(object, cell + ivec3(dx, dy, dz), values[dz][dy][dx]);
}

void get_alphas(map_object, vec3 nx0, vec3 nx1, out float[2][2][2] alphas)
{
    alphas[0][0][0] = 0.125;
    alphas[0][0][1] = 0.125;
    alphas[0][1][0] = 0.125;
    alphas[0][1][1] = 0.125;
    alphas[1][0][0] = 0.125;
    alphas[1][0][1] = 0.125;
    alphas[1][1][0] = 0.125;
    alphas[1][1][1] = 0.125;
}

void add_cell_integral(map_object, float[2][2][2][OUTPUT_DIM] cell, vec3 nx0, vec3 nx1, float dt, inout float _output[OUTPUT_DIM])
{
    float alphas[2][2][2];
    get_alphas(object, nx0, nx1, alphas);

    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] += dt * (
            cell[0][0][0][i] * alphas[0][0][0] + 
            cell[0][0][1][i] * alphas[0][0][1] + 
            cell[0][1][0][i] * alphas[0][1][0] + 
            cell[0][1][1][i] * alphas[0][1][1] + 
            cell[1][0][0][i] * alphas[1][0][0] + 
            cell[1][0][1][i] * alphas[1][0][1] + 
            cell[1][1][0][i] * alphas[1][1][0] + 
            cell[1][1][1][i] * alphas[1][1][1]); 
}

void add_cell_dL_dI(map_object, ivec3 cell, float[OUTPUT_DIM] dL_dI, vec3 nx0, vec3 nx1, float dt)
{
    float alphas [2][2][2];
    get_alphas(object, nx0, nx1, alphas);

    float grads_cell[2][2][2][OUTPUT_DIM];

    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
    {
        float dtdLdI = dt * dL_dI[i];
        grads_cell[0][0][0][i] = dtdLdI * alphas[0][0][0];
        grads_cell[0][0][1][i] = dtdLdI * alphas[0][0][1];
        grads_cell[0][1][0][i] = dtdLdI * alphas[0][1][0];
        grads_cell[0][1][1][i] = dtdLdI * alphas[0][1][1];
        grads_cell[1][0][0][i] = dtdLdI * alphas[1][0][0];
        grads_cell[1][0][1][i] = dtdLdI * alphas[1][0][1];
        grads_cell[1][1][0][i] = dtdLdI * alphas[1][1][0];
        grads_cell[1][1][1][i] = dtdLdI * alphas[1][1][1];
    }
    
    add_grad_cell(object, cell, grads_cell);
}

FORWARD
{
    [[unroll]]
    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0;
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 x_end = vec3(_input[3], _input[4], _input[5]);
    vec3 dx = x_end - x;
    float d = length(dx);
    if (d == 0.0)
        return; // 0 integral value if domain is empty
    vec3 w = dx / d;
    float tMin, tMax;
    if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
        return; // 0 integral value outside bounding box
    x += w * tMin;  
    d = tMax - tMin;
    
    vec3 box_size = parameters.box_max - parameters.box_min;
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 cell_size = box_size / dim;
    ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));
    vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
    ivec3 side = ivec3(sign(w));
    vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
    vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
    float current_t = 0;
    float[2][2][2][OUTPUT_DIM] cell_values;
    
    while(current_t < d - 0.00001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
        
        load_cell(object, cell, cell_values);
        add_cell_integral(object, cell_values, vec3(0.0), vec3(1.0), next_t - current_t, _output); 
        
        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;
    }
}

BACKWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 x_end = vec3(_input[3], _input[4], _input[5]);

    // TODO: Add dL_dx0 and dL_dx1 here... boundary terms
    
    if (parameters.grid.grad_data == 0)
        return; // No grad
        
    vec3 dx = x_end - x;
    float d = length(dx);
    if (d == 0.0)
        return; // 0 integral value if domain is empty

    vec3 w = dx / d;
    float tMin, tMax;
    if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
        return; // 0 integral value outside bounding box
    x += w * tMin;  
    d = tMax - tMin;
    
    vec3 box_size = parameters.box_max - parameters.box_min;
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    vec3 cell_size = box_size / dim;
    ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));
    vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
    ivec3 side = ivec3(sign(w));
    vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
    vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
    float current_t = 0;
    
    while(current_t < d - 0.00001){
        float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
        
        add_cell_dL_dI(object, cell, _output_grad, vec3(0.0), vec3(1.0), next_t - current_t); 
        
        ivec3 cell_inc = ivec3(
            alpha.x <= alpha.y && alpha.x <= alpha.z,
            alpha.x > alpha.y && alpha.y <= alpha.z,
            alpha.x > alpha.z && alpha.y > alpha.z);

        current_t = next_t;
        alpha += cell_inc * alpha_inc;
        cell += cell_inc * side;
    }
}

                """
    )

    def __init__(self, grid_model: Grid3D):
        super(Grid3DLineIntegral, self).__init__(OUTPUT_DIM=grid_model.output_dim)
        self.grid_model = grid_model
        self.grid = grid_model.grid
        self.box_min = grid_model.bmin.clone()
        self.box_max = grid_model.bmax.clone()


class Grid3DTransmittanceDDARayIntegral(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            out_radiance=MapBase,
            boundary_radiance=MapBase,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6),
        code="""

float sigma_at(map_object, ivec3 cell)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    float_ptr buf = param_buffer(parameters.grid, cell);
    return buf.data[0];
}

void load_cell(map_object, ivec3 cell, out float[2][2][2] sigmas)
{
    [[unroll]]
    for (int dz = 0; dz < 2; dz ++)
        [[unroll]]
        for (int dy = 0; dy < 2; dy ++)
            [[unroll]]
            for (int dx = 0; dx < 2; dx ++) 
                sigmas[dz][dy][dx] = sigma_at(object, cell + ivec3(dx, dy, dz));
}

float interpolated_sigma(map_object, vec3 alpha, float[2][2][2] sigmas)
{
    return mix(mix(
        mix(sigmas[0][0][0], sigmas[0][0][1], alpha.x),
        mix(sigmas[0][1][0], sigmas[0][1][1], alpha.x), alpha.y),
        mix(
        mix(sigmas[1][0][0], sigmas[1][0][1], alpha.x),
        mix(sigmas[1][1][0], sigmas[1][1][1], alpha.x), alpha.y), alpha.z);
}

FORWARD
{
    [[unroll]]
    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0;
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    float radiance_values[OUTPUT_DIM];
    float T = 1.0; // full transmittance start

    float tMin, tMax;
    if (intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
    {
        x += w * tMin;  
        float d = tMax - tMin;

        vec3 box_size = parameters.box_max - parameters.box_min;
        ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
        vec3 cell_size = box_size / dim;
        ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
        cell = clamp(cell, ivec3(0), dim - ivec3(1));
        vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
        ivec3 side = ivec3(sign(w));
        vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
        vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
        float current_t = 0;
        vec3 vn = (x - parameters.box_min) * dim / box_size;
        vec3 vm = w * dim / box_size;

        float[2][2][2] sigma_values;

        while(current_t < d - 0.0001){
            float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));

            load_cell(object, cell, sigma_values);
            float cell_t = mix(current_t, next_t, 0.5);
            // ** Accumulate interaction
            // * sample sigma
            vec3 interpolation_alpha = fract(vm * cell_t + vn);
            float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
            float emission_integral = 1 - exp(-sigma_value * (next_t - current_t));
            // if (emission_integral > 0.9) emission_integral = 1.0; else emission_integral = 0.0;
            if (emission_integral > 0.0001){
                vec3 xc = cell_t*w + x;
                forward(parameters.out_radiance, float[6](xc.x, xc.y, xc.z, w.x, w.y, w.z), radiance_values);
                [[unroll]]
                for (int i=0; i<OUTPUT_DIM; i++)
                    _output[i] += T * emission_integral * radiance_values[i];
            }
            T *= (1 - emission_integral);

            if (T < 0.001) break;

            ivec3 cell_inc = ivec3(
                alpha.x <= alpha.y && alpha.x <= alpha.z,
                alpha.x > alpha.y && alpha.y <= alpha.z,
                alpha.x > alpha.z && alpha.y > alpha.z);

            current_t = next_t;
            alpha += cell_inc * alpha_inc;
            cell += cell_inc * side;
        }
        x += w * d;
    }
    forward(parameters.boundary_radiance, float[6](x.x, x.y, x.z, w.x, w.y, w.z), radiance_values);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] += T * radiance_values[i];
}

                """, nodiff=True
    )

    def __init__(self, grid_model: Grid3D, out_radiance: MapBase, boundary_radiance: MapBase):
        super(Grid3DTransmittanceDDARayIntegral, self).__init__(OUTPUT_DIM=out_radiance.output_dim)
        assert out_radiance.output_dim == boundary_radiance.output_dim
        self.grid_model = grid_model
        self.grid = grid_model.grid
        self.out_radiance = out_radiance
        self.boundary_radiance = boundary_radiance
        self.box_min = grid_model.bmin.clone()
        self.box_max = grid_model.bmax.clone()


class Grid3DTransmittanceRayIntegral(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            out_radiance=MapBase,
            boundary_radiance=MapBase,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6),
        code="""

float sigma_at(map_object, ivec3 cell)
{
    ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
    cell = clamp(cell, ivec3(0), dim - ivec3(1));

    float_ptr buf = param_buffer(parameters.grid, cell);
    return buf.data[0];
}

void load_cell(map_object, ivec3 cell, out float[2][2][2] sigmas)
{
    [[unroll]]
    for (int dz = 0; dz < 2; dz ++)
        [[unroll]]
        for (int dy = 0; dy < 2; dy ++)
            [[unroll]]
            for (int dx = 0; dx < 2; dx ++) 
                sigmas[dz][dy][dx] = sigma_at(object, cell + ivec3(dx, dy, dz));
}

float interpolated_sigma(map_object, vec3 alpha, float[2][2][2] sigmas)
{
    return mix(mix(
        mix(sigmas[0][0][0], sigmas[0][0][1], alpha.x),
        mix(sigmas[0][1][0], sigmas[0][1][1], alpha.x), alpha.y),
        mix(
        mix(sigmas[1][0][0], sigmas[1][0][1], alpha.x),
        mix(sigmas[1][1][0], sigmas[1][1][1], alpha.x), alpha.y), alpha.z);
}

FORWARD
{
    [[unroll]]
    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0;
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    
    float radiance_values[OUTPUT_DIM];
    float T = 1.0; // full transmittance start
    
    float tMin, tMax;
    if (intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
    {
        x += w * tMin;  
        float d = tMax - tMin;
    
        vec3 box_size = parameters.box_max - parameters.box_min;
        ivec3 dim = ivec3(parameters.grid.shape[2] - 1, parameters.grid.shape[1] - 1, parameters.grid.shape[0] - 1);
        vec3 cell_size = box_size / dim;
        ivec3 cell = ivec3((x - parameters.box_min) * dim / box_size);
        cell = clamp(cell, ivec3(0), dim - ivec3(1));
        vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
        ivec3 side = ivec3(sign(w));
        vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + parameters.box_min;
        vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
        float current_t = 0;
        vec3 vn = (x - parameters.box_min) * dim / box_size;
        vec3 vm = w * dim / box_size;
        
        float[2][2][2] sigma_values;
    
        while(current_t < d - 0.0001){
            float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
    
            load_cell(object, cell, sigma_values);
            float majorant = max(max(
                max (sigma_values[0][0][0], sigma_values[0][0][1]),
                max (sigma_values[0][1][0], sigma_values[0][1][1])),
                max(
                max (sigma_values[1][0][0], sigma_values[1][0][1]),
                max (sigma_values[1][1][0], sigma_values[1][1][1])));
            
            float cell_t = current_t;
            while (true)
            {
                float dt = -log(1 - random()) / max(0.00001, majorant);
                if (cell_t + dt > next_t)
                break;
                cell_t += dt;
                // ** Accumulate interaction
                // * sample sigma
                vec3 interpolation_alpha = fract(vm * cell_t + vn);
                float sigma_value = interpolated_sigma(object, interpolation_alpha, sigma_values);
                float Pc = min(1.0, sigma_value / majorant);
                vec3 xc = cell_t*w + x;
                forward(parameters.out_radiance, float[6](xc.x, xc.y, xc.z, w.x, w.y, w.z), radiance_values);
                [[unroll]]
                for (int i=0; i<OUTPUT_DIM; i++)
                    _output[i] += T * Pc * radiance_values[i];
                T *= (1 - Pc);
                if (T < 0.001) break;
            }
    
            if (T < 0.001) break;
            
            ivec3 cell_inc = ivec3(
                alpha.x <= alpha.y && alpha.x <= alpha.z,
                alpha.x > alpha.y && alpha.y <= alpha.z,
                alpha.x > alpha.z && alpha.y > alpha.z);
    
            current_t = next_t;
            alpha += cell_inc * alpha_inc;
            cell += cell_inc * side;
        }
        x += w * d;
    }
    forward(parameters.boundary_radiance, float[6](x.x, x.y, x.z, w.x, w.y, w.z), radiance_values);
    [[unroll]]
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] += T * radiance_values[i];
}

                """, nodiff=True
    )

    def __init__(self, grid_model: Grid3D, out_radiance: MapBase, boundary_radiance: MapBase):
        super(Grid3DTransmittanceRayIntegral, self).__init__(OUTPUT_DIM=out_radiance.output_dim)
        assert out_radiance.output_dim == boundary_radiance.output_dim
        self.grid_model = grid_model
        self.grid = grid_model.grid
        self.out_radiance = out_radiance
        self.boundary_radiance = boundary_radiance
        self.box_min = grid_model.bmin.clone()
        self.box_max = grid_model.bmax.clone()


class TransmittanceDDA(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            grid=ParameterDescriptor,
            box_min=vec3,
            box_max=vec3
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        code="""
        
    float get_grid_float(Parameter grid, ivec3 dim, ivec3 cell) {
        // cell = clamp(cell, ivec3(0), dim - 1);
        if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dim)))
            return 0.0f;
        float_ptr voxel_buf = param_buffer(grid, cell);
        return voxel_buf.data[0];
        /* int voxel_index = cell.x + (cell.y + cell.z * grid.dim.y) * grid.dim.x;
        float x[1];
        tensorLoad(grid.ptr, voxel_index, x);
        return x[0]; */
    }    
    
    float sample_grid_float(Parameter grid, ivec3 dim, vec3 p, vec3 b_min, vec3 b_max){
        vec3 fcell = dim * (p - b_min)/(b_max - b_min) - vec3(0.5);
        ivec3 cell = ivec3(floor(fcell));
        vec3 alpha = (fcell - cell);
        float sigma = 0;
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 0, 0))*(1 - alpha.x)*(1 - alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 0, 0))*(alpha.x)*(1 - alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 1, 0))*(1 - alpha.x)*(alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 1, 0))*(alpha.x)*(alpha.y)*(1 - alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 0, 1))*(1 - alpha.x)*(1 - alpha.y)*(alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 0, 1))*(alpha.x)*(1 - alpha.y)*(alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(0, 1, 1))*(1 - alpha.x)*(alpha.y)*(alpha.z);
        sigma += get_grid_float(grid, dim, cell + ivec3(1, 1, 1))*(alpha.x)*(alpha.y)*(alpha.z);
        return sigma;
    }
    
        
    float DDA_Transmittance(Parameter grid, vec3 x, vec3 w, float d, vec3 b_min, vec3 b_max)
    {
        ivec3 dim = ivec3(grid.shape[2] - 1, grid.shape[1] - 1, grid.shape[0] - 1);
        vec3 b_size = b_max - b_min;
        vec3 cell_size = b_size / dim;
        ivec3 cell = ivec3((x - b_min) * dim / b_size);
        cell = clamp(cell, ivec3(0), dim - ivec3(1));
        vec3 alpha_inc = cell_size / max(vec3(0.00001), abs(w));
        ivec3 side = ivec3(sign(w));
        vec3 corner = (cell + side * 0.5 + vec3(0.5)) * cell_size + b_min;
        vec3 alpha = abs(corner - x) / max(vec3(0.00001), abs(w));
        float tau = 0;
        float current_t = 0;
        while(current_t < d - 0.00001){
            float next_t = min(d, min(alpha.x, min(alpha.y, alpha.z)));
            ivec3 cell_inc = ivec3(
                alpha.x <= alpha.y && alpha.x <= alpha.z,
                alpha.x > alpha.y && alpha.y <= alpha.z,
                alpha.x > alpha.z && alpha.y > alpha.z);
            float a = 0.5;//random(seed);
            vec3 xt = x + (next_t*a + current_t*(1-a))*w;
            float voxel_density = sample_grid_float(grid, dim, xt, b_min, b_max);
    
            tau += (next_t - current_t) * voxel_density;
            current_t = next_t;
            alpha += cell_inc * alpha_inc;
            cell += cell_inc * side;
        }
        return exp(-tau);
    }

    FORWARD
    {
        [[unroll]]
        for (int i=0; i < OUTPUT_DIM; i++)
            _output[i] = 0.0;
        vec3 x = vec3(_input[0], _input[1], _input[2]);
        vec3 w = vec3(_input[3], _input[4], _input[5]);
        float tMin, tMax;
        if (!intersect_ray_box(x, w, parameters.box_min, parameters.box_max, tMin, tMax))
            return; // 0 integral value outside bounding box
        x += w * tMin;  
        float d = tMax - tMin;
        float T = DDA_Transmittance (parameters.grid, x, w, d, parameters.box_min, parameters.box_max);
        _output[0] = T;
    }
                    """,
        nodiff=True
    )

    def __init__(self, grid: torch.Tensor, box_min: vec3 = vec3(-1.0, -1.0, -1.0), box_max: vec3 = vec3(1.0, 1.0, 1.0)):
        super(TransmittanceDDA, self).__init__()
        self.grid = torch.nn.Parameter(grid)
        self.box_min = box_min.clone()
        self.box_max = box_max.clone()


class SH_PDF(MapBase):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, SH_DIM=1),
        parameters=dict(
            coefficients=MapBase
        ),
        code="""
    FORWARD
    {
        float Y[SH_DIM];
        eval_sh(vec3(_input[3], _input[4], _input[5]), Y);
        
        float c[SH_DIM * OUTPUT_DIM];
        forward(parameters.coefficients, float[3](_input[0], _input[1], _input[2]), c);
        
        [[unroll]]
        for (int i=0; i<OUTPUT_DIM; i++)
        {
            _output[i] = 0.0;
            [[unroll]]
            for (int j=0; j<SH_DIM; j++)
                _output[i] += Y[j] * c[i*SH_DIM + j];
            _output[i] = max(_output[i], 0.001);
        }
    }

    BACKWARD
    {
        // NOT EASY TASK to backprop input here.
        
        float Y[SH_DIM];
        eval_sh(vec3(_input[3], _input[4], _input[5]), Y);
        
        float dL_dc[SH_DIM * OUTPUT_DIM];
        [[unroll]]
        for (int i=0; i<OUTPUT_DIM; i++)
        {
            [[unroll]]
            for (int j=0; j<SH_DIM; j++)
                dL_dc[i*SH_DIM + j] = _output_grad[i] * Y[j];
        }

        float dL_dx[3];
        backward(parameters.coefficients, float[3](_input[0], _input[1], _input[2]), dL_dc, dL_dx);
        
        [[unroll]]
        for (int i=0; i<3; i++)
            _input_grad[i] += dL_dx[i];
    }
    """,
        nodiff=False,
    )

    def __init__(self, output_dim, coefficients_map: 'MapBase'):
        assert coefficients_map.output_dim % output_dim == 0, f'SH coefficients must divide number of channels {output_dim}'
        SH_DIM = coefficients_map.output_dim // output_dim
        assert SH_DIM in [1, 4, 9], 'Not supported higher order SH than 3'
        assert coefficients_map.input_dim == 3
        super(SH_PDF, self).__init__(OUTPUT_DIM=output_dim, SH_DIM=SH_DIM)
        self.coefficients = coefficients_map


class RayBoxIntersection(MapBase):
    __extension_info__ = dict(
        generics=dict(
            INPUT_DIM=6,
            OUTPUT_DIM=2
        ),
        parameters=dict(
            box_min=vec3,
            box_max=vec3
        ),
        code="""
FORWARD
{
    ray_box_intersection(
        vec3(_input[0], _input[1], _input[2]), 
        vec3(_input[3], _input[4], _input[5]), 
        parameters.box_min, 
        parameters.box_max, 
        _output[0], _output[1]);
}
        """,
        nodiff=True
    )

    def __init__(self, box_min:vec3, box_max: vec3, **kwargs):
        super(RayBoxIntersection, self).__init__()
        self.box_min = box_min
        self.box_max = box_max


class VolumeRadianceIntegratorBase(MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(radiance_code: str, *requires: str, **parameters):
        return dict(
            generics=dict(
                INPUT_DIM=6,
                OUTPUT_DIM=3,
                **{ 'VR_'+k.upper(): 1 for k in requires }
            ),
            parameters={ **{k: MapBase for k in requires}, **parameters },
            code = f"""
#include "common_vr.h"

{radiance_code}

FORWARD {{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 R = volume_radiance(object, x, w);
    _output = float[3](R.x, R.y, R.z);
}}

BACKWARD {{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 dL_dR = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    volume_radiance_bw(object, x, w, dL_dR);
}}
            """
        )

    def __init__(self, **maps):
        required_maps = type(self).__extension_info__['parameters'].keys()
        assert all(r in required_maps for r in maps)
        assert all(r in maps for r in required_maps)
        assert all(v is None or isinstance(v, MapBase) for v in maps.values())
        super(VolumeRadianceIntegratorBase, self).__init__()
        for k,v in maps.items():
            setattr(self, k, v)


class DTCollisionIntegrator(VolumeRadianceIntegratorBase):
    '''
    I(x, w) = \int_0^d(x,w) T(x_0, x_t) \sigma(x_t) F(x_t, w) dt + T(x_0, x_d) B(x_t, w)
    '''
    __extension_info__ = VolumeRadianceIntegratorBase.create_extension_info(
"""
vec3 volume_radiance(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    if (boundary(object, x, w, tMin, tMax))
    {
        x += w * tMin;
        tMax -= tMin;
        vec3 xd = x + w * tMax;
        tMax -= empty_space(object, xd, -w);
        tMin = empty_space(object, x, w);
        x += w * tMin;
        float d = tMax - tMin;
                 
        while (d > 0.00000001)
        {
            float md;
            float maj = majorant(object, x, w, md);
            float local_d = min(d, md);

            float t = -log(1 - random())/maj;

            x += min(t, local_d) * w;

            if (t > local_d)
            {
                d -= local_d;
                continue;
            }

            if (random() < sigma(object, x) / maj)
                return out_radiance(object, x, -w);

            d -= t;
        }
    }
    
    return boundary_radiance(object, x, w);
}

void volume_radiance_bw(map_object, vec3 x, vec3 w, vec3 dL_dR) {

}
        """,
        "sigma", "majorant", "boundary", "empty_space", "out_radiance", "boundary_radiance"
    )
#         code = """
# #include "common_vr.h"
#
# FORWARD
# {
#     vec3 x = vec3(_input[0], _input[1], _input[2]);
#     vec3 w = vec3(_input[3], _input[4], _input[5]);
#     // Get boundary
#     float tMin, tMax;
#     float adding_radiance[OUTPUT_DIM];
#     if (ray_volume_intersection(object, x, w, tMin, tMax))
#     {
#         x += w * tMin;
#         float d = tMax - tMin;
#         float t = ray_empty_space(object, x, w);
#         x += w * t; // advance to first non-zero density
#         d -= t;
#         while (d > 0)
#         {
#             float md ;
#             float majorant = ray_majorant(object, x, w, md);
#             float local_d = min(d, md);
#
#             t = -log(1 - random())/majorant;
#
#             x += min(t, local_d) * w;
#
#             if (t > local_d)
#             {
#                 d -= local_d;
#                 continue;
#             }
#
#             if (random() < sigma(object, x) / majorant)
#             {
#                 out_radiance(object, x, -w, adding_radiance);
#                 [[unroll]] for (int i=0; i<OUTPUT_DIM; i++)
#                     _output[i] = adding_radiance[i];
#                 return;
#             }
#
#             d -= t;
#         }
#     }
#     boundary_radiance(object, x, w, adding_radiance);
#     [[unroll]] for (int i=0; i<OUTPUT_DIM; i++)
#         _output[i] = adding_radiance[i];
# }
#         """, nodiff=True

    def __init__(self,
                 sigma: MapBase,
                 boundary: MapBase,
                 empty_space: MapBase,
                 majorant: MapBase,
                 out_radiance: MapBase,
                 boundary_radiance: MapBase
                 ):
        assert out_radiance.output_dim == boundary_radiance.output_dim
        super(DTCollisionIntegrator, self).__init__(sigma=sigma, boundary=boundary, empty_space=empty_space, majorant=majorant, out_radiance=out_radiance, boundary_radiance=boundary_radiance)


class DTRadianceIntegrator(VolumeRadianceIntegratorBase):
    __extension_info__ = VolumeRadianceIntegratorBase.create_extension_info(
        """
vec3 volume_radiance(map_object, vec3 x, vec3 w)
{
    float tMin, tMax;
    if (!boundary(object, x, w, tMin, tMax))
        return environment(object, w);
    
    vec3 W = vec3(1.0); // importance of the path
    vec3 A = vec3(0.0); // radiance accumulation
    
    x += w * tMin;
    tMax -= tMin;
    vec3 xd = x + w * tMax;
    tMax -= empty_space(object, xd, -w);
    tMin = empty_space(object, x, w);
    x += w * tMin;
    float d = tMax - tMin;
             
    bool some_collision = false;
             
    while (d > 0.00000001)
    {
        float md;
        float maj = majorant(object, x, w, md);
        float local_d = min(d, md);

        float t = -log(1 - random())/maj;

        x += min(t, local_d) * w;

        if (t > local_d)
        {
            d -= local_d;
            continue;
        }
        
        if (random() < sigma(object, x) / maj)
        { 
            // ** Accumulate collision contribution, scatter and continue
            vec3 s = scattering_albedo(object, x);
            vec3 a = vec3(1.0) - s;
            // add emitted radiance
            A += W * a * emission(object, x); 
            // add NEE
            W *= s;
            vec3 w_wenv;
            vec3 wenv = environment_sampler(object, x, w, w_wenv);
            float nee_T = transmittance(object, x, x + wenv * neetMax); 
            A += W * nee_T * phase(object, x, w, wenv) * w_wenv;
            // continue with indirect contribution
            float w_wph;
            vec3 wph = phase_sampler(object, x, w, w_wph);
            w = wph;
            W *= w_wph;
            
            boundary(object, x, w, tMin, tMax);
            x += w * tMin; 
            d = tMax - tMin;
            some_collision = true;
            continue;
        }
        
        // Null collide
        d -= t;
    }
    
    if (!some_collision)
        A += W * environment(object, x, w); 
    
    return A;
}
        """,
        "sigma", "boundary", "empty_space", "majorant", "scattering_albedo", "emission", "environment",
        "phase", "environment_sampler", "phase_sampler", "transmittance"
    )

    def __init__(self,
                 sigma: MapBase,
                 boundary: MapBase,
                 empty_space: MapBase,
                 majorant: MapBase,
                 scattering_albedo: MapBase,
                 emission: MapBase,
                 environment: MapBase,
                 phase: MapBase,
                 environment_sampler: MapBase,
                 phase_sampler: MapBase,
                 transmittance: MapBase
                 ):
        assert emission.output_dim == environment.output_dim == scattering_albedo.output_dim == 3
        super(DTRadianceIntegrator, self).__init__(
            sigma=sigma,
            boundary=boundary,
            empty_space=empty_space,
            majorant=majorant,
            scattering_albedo=scattering_albedo,
            emission=emission,
            environment=environment,
            phase=phase,
            environment_sampler=environment_sampler,
            phase_sampler=phase_sampler,
            transmittance=transmittance
        )

# class RTCollisionIntegrator(VolumeRadianceIntegratorBase):



# class Concat(MapBase):
#     __extension_info__ = dict(
#         parameters=dict(
#             map_a = MapBase,
#             map_b = MapBase
#         )
#     )
#     def __init__(self, map_a: MapBase, map_b: MapBase):
#         assert map_a.input_dim == map_b.input_dim
#         super(Concat, self).__init__(INPUT_DIM = map_a.input_dim, OUTPUT_DIM = map_a.output_dim + map_b.output_dim)


class HGPhase(MapBase):
    __extension_info__ = dict(
        generics=dict(
            INPUT_DIM=9,
            OUTPUT_DIM=1,
        ),
        parameters=dict(
            phase_g=MapBase,
        ),
        code = """
FORWARD
{
    float _g[1];
    forward(parameters.phase_g, float[3](_input[0], _input[1], _input[2]), _g);
    float g = _g[0];    
    _output[0] = hg_phase_eval(vec3(_input[3], _input[4], _input[5]), vec3(_input[6], _input[7], _input[8]), g);
}
        """, nodiff=True
    )

    def __init__(self, phase_g: MapBase):
        super(HGPhase, self).__init__()
        self.phase_g = phase_g


class HGPhaseSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            phase_g=MapBase
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=4),
        code="""
FORWARD
{
    float g[1];
    forward(parameters.phase_g, float[3](_input[0], _input[1], _input[2]), g);
    
    vec3 w_out = hg_phase_sample(vec3(_input[3], _input[4], _input[5]), g[0]);
    _output = float[4](1.0, w_out.x, w_out.y, w_out.z);
}
        """, nodiff=True
    )
    def __init__(self, phase_g: MapBase):
        super(HGPhaseSampler, self).__init__()
        self.phase_g = phase_g


class HGDirectionSampler(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            phase_g=MapBase
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=4),
        code="""
FORWARD
{
    float g[1];
    forward(parameters.phase_g, float[3](_input[0], _input[1], _input[2]), g);
    vec3 w_in = vec3(_input[3], _input[4], _input[5]);
    vec3 w_out = hg_phase_sample(w_in, g[0]);
    _output = float[4](w_out.x, w_out.y, w_out.z, 1.0/hg_phase_eval(w_in, w_out, g[0]));
}
        """, nodiff=True
    )

    def __init__(self, phase_g: MapBase):
        super(HGDirectionSampler, self).__init__()
        self.phase_g = phase_g


__endline__ = '\n'


def map_to_generic(map_name: str):
    return 'VR_'+map_name.upper()


# class VolumeRadianceFieldBase(MapBase):
#     __extension_info__ = None  # abstract node
#
#     @staticmethod
#     def create_extension_info(compute_radiance_code, *maps):
#         return dict(
#             generics=dict(
#                 INPUT_DIM=6,
#                 OUTPUT_DIM=3,
#             ),
#             parameters=dict(
#                 **{k: MapBase for k in maps },
#                 box_min=vec3,
#                 box_max=vec3
#             ),
#             code = f"""
# #include "common_vr.h"
#
# void compute_radiance(map_object, vec3 x, vec3 w, out vec3 wout, out float A[OUTPUT_DIM], out float W[OUTPUT_DIM]);
#
# FORWARD
# {{
#     vec3 x = vec3(_input[0], _input[1], _input[2]);
#     vec3 w = vec3(_input[3], _input[4], _input[5]);
#     float W[OUTPUT_DIM];
#     vec3 ws;
#     compute_radiance(object, x, w, ws, _output, W); // it is missing to retrieve proper jaccobians for x,w -> ws
#     float env[OUTPUT_DIM];
#     environment(object, ws, env);
#     [[unroll]] for (int i = 0; i < OUTPUT_DIM; i++) _output[i] += W[i] * env[i];
# }}
#
# // TODO: the jaccobian for the ray transitions is missing. This can not propagate gradients wrt camera rays
# void compute_radiance_bw(map_object, vec3 x, vec3 w, float o_A[OUTPUT_DIM], float o_W[OUTPUT_DIM], float dL_dA[OUTPUT_DIM], float dL_dW[OUTPUT_DIM]);
#
# BACKWARD
# {{
#     float[OUTPUT_DIM] dL_dA = _output_grad;
#
#     uvec4 seed = get_seed(); // save current seed for later replay
#
#     vec3 x = vec3(_input[0], _input[1], _input[2]);
#     vec3 w = vec3(_input[3], _input[4], _input[5]);
#     float A[OUTPUT_DIM];
#     float W[OUTPUT_DIM];
#     vec3 ws;
#     compute_radiance(object, x, w, ws, A, W);
#
#     //A += W * environment(object, ws);
#
#     float dL_dW[OUTPUT_DIM];
#     float env[OUTPUT_DIM];
#     environment(object, ws, env);
#     [[unroll]] for (int i=0; i<OUTPUT_DIM; i++) dL_dW[i] = dL_dA[i] * env[i];
#
#     //vec3 dL_denv = dL_dA * W;
#     //vec3 dL_dw;
#     //environment_bw(object, ws, dL_denv, dL_dw);
#
#     set_seed(seed); // start replaying
#
#     // precomputed output values
#     compute_radiance_bw(object, x, w, A, W, dL_dA, dL_dW);
#
#     // TODO: in a future when ray jaccobian is considered input_grad can be updated
# }}
#
#
# """ + compute_radiance_code
#         )
#
#     def __init__(self, box_min:vec3, box_max: vec3, **kwargs):
#         super(VolumeRadianceFieldBase, self).__init__(**{map_to_generic(k): v.output_dim for k,v in kwargs.items()})
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#         self.box_min = box_min
#         self.box_max = box_max
#
#     def intersect_ray_box(self, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         w = torch.where(torch.abs(w) <= 0.000001, torch.full_like(w, fill_value=0.000001), w)
#         C_Min = (self.box_min.to(x.device) - x)/w
#         C_Max = (self.box_max.to(x.device) - x)/w
#         min_C = torch.minimum(C_Min, C_Max)
#         max_C = torch.maximum(C_Min, C_Max)
#         tMin = torch.clamp_min(torch.max(min_C, dim=-1, keepdim=True)[0], 0.0)
#         tMax = torch.min(max_C, dim=-1, keepdim=True)[0]
#         return (tMax > tMin)*(tMax > 0), tMin, tMax
#
#     def ray_enter(self, x: torch.Tensor, w: torch.Tensor):
#         mask, tMin, tMax = self.intersect_ray_box(x, w)
#         return mask, x + w*tMin
#
#     def ray_exit(self, x: torch.Tensor, w: torch.Tensor):
#         mask, tMin, tMax = self.intersect_ray_box(x, w)
#         return tMax
#
#     def compute_radiance_torch(self, alive: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         pass
#
#     def forward_torch(self, *args):
#         xw, = args
#         x = xw[...,0:3]
#         w = xw[...,3:6]
#         A = torch.zeros_like(x)
#         W = torch.ones_like(x)
#         entered, x = self.ray_enter(x, w)
#         Av, Wv = self.compute_radiance_torch(entered, x, w)
#         W = torch.where(entered, Wv, W)
#         A = torch.where(entered, Av, A)
#         A += W * self.environment(w)
#         return A
#

# class AbsorptionOnlyVolume(VolumeRadianceFieldBase):
#     __extension_info__ = VolumeRadianceFieldBase.create_extension_info("""
# float compute_tau(map_object, inout vec3 x, vec3 w, float d)
# {
#     float total_sigma = 0;
#     float dt = 0.005;
#     int samples = int(d / dt) + 1;
#     vec3 dw = w * dt;
#     x += dw * 0.5;
#     [[unroll]]
#     for (int i=0; i<samples; i++)
#     {
#         total_sigma += sigma(object, x);
#         x += dw;
#     }
#     return total_sigma * d / samples;
# }
#
# void compute_radiance(map_object, inout vec3 x, inout vec3 w, out vec3 A, out vec3 W)
# {
#     A = vec3(0.0);
#     float d = ray_exit(object, x, w);
#     float tau = compute_tau(object, x, w, d);
#     W = vec3(1.0) * exp(-tau);
# }
#     """, ['sigma', 'environment'])
#
#     def __init__(self, sigma: 'MapBase', environment: 'MapBase', box_min:vec3, box_max: vec3):
#         super(AbsorptionOnlyVolume, self).__init__(box_min, box_max, sigma = sigma, environment = environment)
#
#     def compute_tau(self, alive: torch.Tensor, x: torch.Tensor, w: torch.Tensor, d: torch.Tensor):
#         tau = torch.zeros(*x.shape[:-1], 1, device=x.device)
#         i = torch.zeros(*x.shape[:-1], 1, device=x.device)
#         dt = 0.005
#         samples = (d / dt).int() + 1
#         dw = w * dt
#         x += dw * 0.5
#
#         while alive.any():
#             tau += self.sigma(x)
#             x += dw
#             i += 1
#             alive *= i < samples
#
#         return tau * d / samples
#
#     def compute_radiance_torch(self, alive: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         A = vec3(0.0);
#         float d = ray_exit(x, w);
#         W = vec3(1.0) * exp(-d);
#         """
#         d = self.ray_exit(x, w)
#         A = torch.zeros_like(x)
#         W = torch.ones_like(x) * torch.exp(-self.compute_tau(alive.clone(), x, w, d))
#         return A, W
#

class AbsorptionOnlyVolume(VolumeRadianceIntegratorBase):
    __extension_info__ = VolumeRadianceIntegratorBase.create_extension_info("""
vec3 volume_radiance(map_object, vec3 x, vec3 w)
{
    float T = transmittance(object, x, x + w*INF_DISTANCE);
    vec3 R = T * environment(object, w);
    return R;
}

void volume_radiance_bw(map_object, vec3 x, vec3 w, vec3 dL_dR)
{
    float dL_dT = dot(dL_dR, environment(object, w));
    transmittance_bw(object, x, x + w * INF_DISTANCE, dL_dT);
}
    """, 'transmittance', 'environment')

    def __init__(self, transmittance: 'MapBase', environment: 'MapBase'):
        super(AbsorptionOnlyVolume, self).__init__(transmittance=transmittance, environment = environment)





#
# class AbsorptionOnlyXVolume(VolumeRadianceFieldXBase):
#     __extension_info__ = dict(
#         path=__DISPATCHING_FOLDER__ + '/vrf/ao_tensor',
#         nodiff=True,
#         supports_express=True,
#         force_compilation=True,
#         parameters=Layout.create_structure('scalar',
#                                            sigma=torch.int64,
#                                            sigma_shape=ivec3,
#                                            environment=torch.int64,
#                                            environment_shape=ivec2,
#                                            box_min=vec3,
#                                            box_max=vec3,
#                                            custom=dict( foo=int )
#                                            )
#     )
#
#     def __init__(self, sigma: torch.Tensor, environment: torch.Tensor, box_min:vec3, box_max: vec3):
#         super(AbsorptionOnlyXVolume, self).__init__(sigma, environment, box_min, box_max)
#
#     def compute_tau(self, alive: torch.Tensor, x: torch.Tensor, w: torch.Tensor, d: torch.Tensor):
#         tau = torch.zeros(*x.shape[:-1], 1, device=x.device)
#         i = torch.zeros(*x.shape[:-1], 1, device=x.device)
#         dt = 0.005
#         samples = (d / dt).int() + 1
#         dw = w * dt
#         x += dw * 0.5
#
#         while alive.any():
#             tau += self.sigma(x)
#             x += dw
#             i += 1
#             alive *= i < samples
#
#         return tau * d / samples
#
#     def compute_radiance_torch(self, alive: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         A = vec3(0.0);
#         float d = ray_exit(x, w);
#         W = vec3(1.0) * exp(-d);
#         """
#         d = self.ray_exit(x, w)
#         A = torch.zeros_like(x)
#         W = torch.ones_like(x) * torch.exp(-self.compute_tau(alive.clone(), x, w, d))
#         return A, W


def normalized_box(s: Union[torch.Tensor, List]) -> Tuple[vec3, vec3]:
    if isinstance(s, torch.Tensor):
        shape = s.shape
    else:
        shape = s
    max_dim = max(shape[0], shape[1], shape[2]) - 1
    b_max : vec3 = vec3(shape[2] - 1, shape[1] - 1, shape[0] - 1) * 0.5 / max_dim
    b_min = -b_max
    return (b_min, b_max)

def zero(dim):
    raise NotImplementedError()

def one(dim):
    raise NotImplementedError()

def constant(input_dim, *args):
    return ConstantMap(input_dim=input_dim, value=torch.tensor([*args], device=device(), dtype=torch.float))

def ray_to_segment(distance_field: 'MapBase'):
    return RayToSegment(distance_field)

def grid2d(t: torch.Tensor, bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
    return Grid2D(t, bmin, bmax)

def image2d(t: torch.Tensor, bmin: vec2 = vec2(-1.0, -1.0), bmax: vec2 = vec2(1.0, 1.0)):
    return Image2D(t, bmin, bmax)

def grid3d(t: torch.Tensor, bmin: vec3 = vec3(-1.0, -1.0, -1.0), bmax: vec3 = vec3(1.0, 1.0, 1.0)):
    return Grid3D(t, bmin, bmax)

def transmittance(sigma: 'MapBase', majorant: float = None, mode: Literal['dt', 'rt'] = 'rt'):
    if majorant is None:
        assert isinstance(sigma, Grid3D)
        majorant = sigma.grid.max().item()
    if mode == 'rt':
        return TransmittanceRT(sigma, majorant)
    if mode == 'dt':
        return TransmittanceDT(sigma, majorant)
    raise Exception()

def xr_projection(ray_input: bool = False):
    return XRProjection(ray_input=ray_input)

def tsr(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def tsr_normal(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def tsr_position_normal(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def tsr_ray(cls, translate: vec3 = vec3(0.0, 0.0, 0.0), scale: vec3 = vec3(1.0, 1.0, 1.0), rotatation_axis: vec3 = vec3(0.0, 1.0, 0.0), rotation_angle: float = 0.0):
    raise NotImplementedError()

def spherical_projection(cls):
    raise NotImplementedError()

def cylindrical_projection(cls):
    raise NotImplementedError()

def identity(input_dim: int):
    return Identity(input_dim)

def ray_position():
    return RayPosition()

def ray_direction():
    return RayDirection()

def ray_box_intersection(bmin: vec3, bmax: vec3):
    return RayBoxIntersection(box_min=bmin, box_max=bmax)