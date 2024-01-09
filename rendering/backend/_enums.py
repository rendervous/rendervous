from enum import IntEnum


class PresenterMode(IntEnum):
    NONE = 0
    OFFLINE = 1
    WINDOW = 2


class QueueType(IntEnum):
    NONE = 0
    COPY = 1
    COMPUTE = 2
    GRAPHICS = 3
    RAYTRACING = 4


class BufferUsage(IntEnum):
    NONE = 0
    STAGING = 1
    VERTEX = 2
    INDEX = 3
    UNIFORM = 4
    STORAGE = 5
    RAYTRACING_ADS = 6
    RAYTRACING_RESOURCE = 7
    SHADER_TABLE = 8


class ImageUsage(IntEnum):
    NONE = 0
    TRANSFER = 1
    SAMPLED = 2
    RENDER_TARGET = 3
    DEPTH_STENCIL = 4
    STORAGE = 5
    ANY = 6


class Format(IntEnum):
    NONE = 0
    UINT_RGBA = 1
    UINT_RGB = 2
    UINT_BGRA_STD = 3
    UINT_RGBA_STD = 4
    UINT_RGBA_UNORM = 5
    UINT_BGRA_UNORM = 6
    FLOAT = 7
    INT = 8
    UINT = 9
    VEC2 = 10
    VEC3 = 11
    VEC4 = 12
    IVEC2 = 13
    IVEC3 = 14
    IVEC4 = 15
    UVEC2 = 16
    UVEC3 = 17
    UVEC4 = 18
    PRESENTER = 19


class ImageType(IntEnum):
    NONE = 0
    TEXTURE_1D = 1
    TEXTURE_2D = 2
    TEXTURE_3D = 3


class MemoryLocation(IntEnum):
    """
    Memory configurations
    """
    NONE = 0
    """
    Efficient memory for reading and writing on the GPU.
    """
    GPU = 1
    """
    Memory can be read and write directly from the CPU
    """
    CPU = 2


class ShaderStage(IntEnum):
    NONE = 0
    VERTEX = 1
    FRAGMENT = 2
    COMPUTE = 3
    RT_GENERATION = 4
    RT_CLOSEST_HIT = 5
    RT_MISS = 6
    RT_ANY_HIT = 7
    RT_INTERSECTION_HIT = 8
    RT_CALLABLE = 9


class Filter(IntEnum):
    NONE = 0
    POINT = 1
    LINEAR = 2


class MipMapMode(IntEnum):
    NONE = 0
    POINT = 1
    LINEAR = 2


class AddressMode(IntEnum):
    NONE = 0
    REPEAT = 1
    CLAMP_EDGE = 2
    BORDER_COLOR = 3


class CompareOp(IntEnum):
    NONE = 0
    NEVER = 1
    LESS = 2
    EQUAL = 3
    LESS_OR_EQUAL = 4
    GREATER = 5
    NOT_EQUAL = 6
    GREATER_OR_EQUAL = 7
    ALWAYS = 8


class BorderColor(IntEnum):
    NONE = 0
    TRANSPARENT_BLACK_FLOAT = 1
    TRANSPARENT_BLACK_INT = 2
    OPAQUE_BLACK_FLOAT = 3
    OPAQUE_BLACK_INT = 4
    OPAQUE_WHITE_FLOAT = 5
    OPAQUE_WHITE_INT = 6


class ADSNodeType(IntEnum):
    NONE = 0
    TRIANGLES = 1
    AABB = 2
    INSTANCE = 3


class PipelineType(IntEnum):
    NONE = 0
    COMPUTE = 1
    GRAPHICS = 2
    RAYTRACING = 3


class DescriptorType(IntEnum):
    NONE = 0
    SAMPLER = 1
    UNIFORM_BUFFER = 2
    STORAGE_BUFFER = 3
    STORAGE_IMAGE = 4
    SAMPLED_IMAGE = 5
    COMBINED_IMAGE = 6
    SCENE_ADS = 7


