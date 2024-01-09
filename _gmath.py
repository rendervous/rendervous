import torch
import math
from typing import Any


def _tensify(t, dimension):
    """
    Tool method to convert a torch tensor into a graphic tensor or scalar
    """
    assert dimension >= 0, "Dimension can not be negative"
    if dimension == 0 or dimension > 2:  #scalar or multidimensional array
        return t
    assert len(t.shape) >= dimension, "can not create a matrix from a vector tensor"
    if tuple(t.shape[-dimension:]) not in __SHAPE_TO_TYPE__:
        return t.as_subclass(torch.Tensor)
    typ = __SHAPE_TO_TYPE__[t.shape[-dimension:]]
    if t.dtype == torch.int32:
        if typ == 'float':
            typ = 'int'
        else:
            typ = 'i'+typ
    typ = eval(typ)
    return typ(t)


def _tensify_batch(t, batch_shape):
    if not isinstance(t, torch.Tensor):
        return t
    if len(t.shape) < len(batch_shape) or list(t.shape[:len(batch_shape)]) != list(batch_shape):
        return t.as_subclass(torch.Tensor)  # is not batched
    dimension = len(t.shape) - len(batch_shape)
    return _tensify(t, dimension)


class GTensorMeta(torch._C._TensorMeta):
    def __new__(mcs, name, bases, dct):
        tensor_type = super().__new__(mcs, name, bases, dct)
        if name.startswith('i'):
            dtype = torch.int32
            name = name[1:]
        else:
            dtype = torch.float32
        dict = {
            'vec1': (1,),
            'vec2': (2,),
            'vec3': (3,),
            'vec4': (4,),
            'mat1': (1,1),
            'mat2': (2,2),
            'mat3': (3,3),
            'mat4': (4,4),
        }
        if name not in dict:
            return tensor_type
        tensor_type.tensor_shape = dict[name]
        tensor_type.tensor_dtype = dtype
        tensor_type.dimension = len(tensor_type.tensor_shape)
        return tensor_type

    def __promote_constant(self, value):
        if not isinstance(value, torch.Tensor):
            return torch.full(self.tensor_shape, fill_value=value, dtype=self.tensor_dtype)
        return value.repeat()

    @staticmethod
    def _check_componentwise_tensors(args):
        if not isinstance(args, list):
            return False
        if len(args) == 0:
            return False
        for a in args:
            if not isinstance(a, torch.Tensor):
                return False
            if a.shape != args[0].shape:
                return False
        return True

    def __call__(cls, *args, **kwargs):
        if len(args) == 0:
            args = [0.0]
        if len(args) == 1:  # promote
            if not isinstance(args[0], torch.Tensor):
                if isinstance(args[0], list):
                    args = torch.as_tensor(args[0], dtype=cls.tensor_dtype)
                    assert list(args.shape[-cls.dimension:]) == list(cls.tensor_shape), "Error with shapes in list elements"
                else:
                    # Assume is scalar
                    args = cls.__promote_constant(args[0])
            else:
                # Check if it is a scalar batched tensor
                args = args[0].type(cls.tensor_dtype)
                if args.shape[-1] == 1: # need to promote
                    args = args.repeat(*tuple([1]*len(args.shape[:-1])), math.prod(cls.tensor_shape))
                    args = args.view(*args.shape[:-1], *cls.tensor_shape)
                assert args.shape[-cls.dimension:] == cls.tensor_shape, f"Wrong shape for tensor argument, expected final dimension to be 1 or {cls.tensor_shape}"
        if not isinstance(args, torch.Tensor):
            if _GTensorBase._check_componentwise_tensors(args):
                args = torch.cat(args, dim=-1).type(cls.tensor_dtype)
            else:
                args = torch.as_tensor([*args], dtype=cls.tensor_dtype).view(cls.tensor_shape)
        assert args.shape[-cls.dimension:] == cls.tensor_shape, f'Wrong vector dimension, expected {cls.shape} provided {args.shape[-cls.dimension:]}'
        tensor_instance = super(GTensorMeta, cls).__call__(args)
        tensor_instance = tensor_instance.as_subclass(cls)
        batch_dimension = len(args.shape) - cls.dimension
        object.__setattr__(tensor_instance, 'batch_dimension', batch_dimension)
        object.__setattr__(tensor_instance, 'batch_shape', args.shape[:batch_dimension])
        object.__setattr__(tensor_instance, 'is_batch', batch_dimension > 0)
        return tensor_instance


__SHAPE_TO_TYPE__ = {
    # (1,): 'vec1',
    (2,): 'vec2',
    (3,): 'vec3',
    (4,): 'vec4',
    # (1, 1): 'mat1',
    (2, 2): 'mat2',
    (3, 3): 'mat3',
    (4, 4): 'mat4'
}


__FIELDS_INDEX__ = {
    'x': 0,
    'y': 1,
    'z': 2,
    'w': 3,
}


class _GTensorBase(torch.Tensor, metaclass=GTensorMeta):
    def __init__(self, *args):
        super(_GTensorBase, self).__init__()
        batch_dimension = len(self.shape) - self.dimension
        object.__setattr__(self, 'batch_dimension', batch_dimension)
        object.__setattr__(self, 'batch_shape', self.shape[:batch_dimension])
        object.__setattr__(self, 'is_batch', batch_dimension > 0)

    @classmethod
    def length(cls, x):
        return _tensify_batch(torch.sqrt((x**2).sum(dim=-1, keepdim=True)), x.batch_shape)

    @classmethod
    def dot(cls, a, b):
        return (a*b).sum(-1, keepdim=True)

    @classmethod
    def normalize(self, v):
        return v / torch.sqrt(_GTensorBase.dot(v, v))  #.detach()

    @classmethod
    def identity(cls):
        assert cls.dimension == 2, "Identity function only valid for matrices"
        assert cls.tensor_shape[0] == cls.tensor_shape[1], "Identity function only valid for squared matrices"
        return cls(torch.diag(torch.full(cls.tensor_shape[0:1], 1, dtype=cls.tensor_dtype)))

    @classmethod
    def zero(cls):
        return cls(torch.zeros(cls.tensor_shape, dtype=cls.tensor_dtype))

    @classmethod
    def one(cls):
        return cls(torch.ones(cls.tensor_shape, dtype=cls.tensor_dtype))

    def __repr__(self):
        return super(_GTensorBase, self).__repr__().replace("tensor", self.__class__.__name__)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunction():
            if func == torch.Tensor.as_subclass:
                return func(*args, **kwargs)
            if func == torch.Tensor.requires_grad_:
                return func(*args, **kwargs)
            if kwargs is None:
                kwargs = {}
            if func.__name__ == '__get__':
                return func(*args, **kwargs)
            batch_shape = None
            for tensor in args:
                if isinstance(tensor, _GTensorBase) and tensor.is_batch:
                    current_batch_shape = tensor.shape[:tensor.batch_dimension]
                    if batch_shape is None:
                        batch_shape = current_batch_shape
                    else:
                        assert batch_shape == current_batch_shape, "Can not operate two graphic tensors with different batches"
            if batch_shape is None:
                batch_shape = []
            if kwargs is None:
                kwargs = {}
            args = tuple(a if not isinstance(a, _GTensorBase) else a.as_subclass(torch.Tensor) for a in args)  # retrieve all tensors from GTensors
            kwargs = {k: v if not isinstance(v, _GTensorBase) else v.as_subclass(torch.Tensor) for k, v in kwargs.items()}

            ret = func(*args, **kwargs)
            return _tensify_batch(ret, batch_shape)

    def __getitem__(self, item):
        last_indexes = [*item] if isinstance(item, tuple) else [item]
        full_index = [slice(None)]*self.batch_dimension + last_indexes
        t = super(_GTensorBase, self).__getitem__(full_index)
        return _tensify(t, len(t.shape) - self.batch_dimension)

    def __setitem__(self, item, value):
        last_indexes = [*item] if isinstance(item, tuple) else [item]
        full_index = [slice(None)]*self.batch_dimension + last_indexes
        super(_GTensorBase, self).__setitem__(full_index, value)

    def __iter__(self):
        assert not self.is_batch, "Not supported yet iteration over tensor batch"
        return super(_GTensorBase, self).__iter__()

    def __getattr__(self, item):
        if item in __FIELDS_INDEX__:
            return self[__FIELDS_INDEX__[item]]
        if all(c in __FIELDS_INDEX__ for c in item):  # all are fields
            index = [__FIELDS_INDEX__[c] for c in item]
            return self[index]
        try:
            return super(_GTensorBase, self).__getattr__(item)
        except:
            pass
        return super(_GTensorBase, self).__getattribute__(item)
    
    def __setattr__(self, key, value):
        if key in __FIELDS_INDEX__:
            self[__FIELDS_INDEX__[key]] = value
            return
        if all(c in __FIELDS_INDEX__ for c in key):  # all are fields
            index = [__FIELDS_INDEX__[c] for c in key]
            self[index] = value
            return
        super(_GTensorBase, self).__setattr__(key, value)


# class vec1(_GTensorBase):
#     pass


class vec2(_GTensorBase):
    pass


class vec3(_GTensorBase):
    @classmethod
    def cross(cls, a, b):
        return vec3(torch.cross(a, b))
        # return vec3(a.y*b.x - a.x*b.y, a.x*b.z - a.z*b.x, a.y*b.z - a.z*b.y)


class vec4(_GTensorBase):
    pass


# class ivec1(_GTensorBase):
#     pass


class ivec2(_GTensorBase):
    pass


class ivec3(_GTensorBase):
    pass


class ivec4(_GTensorBase):
    pass


# class mat1(_GTensorBase):
#     pass


class mat2(_GTensorBase):
    pass


class mat3(_GTensorBase):
    pass


# class LookAtAutograd(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, *args: Any) -> Any:
#         ori, dir, nor = args
#         ctx.save_for_backward(*args)
#         zaxis = dir
#         xaxis = vec3.normalize(vec3.cross(nor, zaxis))
#         yaxis = vec3.cross(zaxis, xaxis)
#         result = mat4(torch.zeros(*ori.shape[:-1], 4, 4, dtype=torch.float32, device=ori.device))
#         result[:, 0:3, 0] = xaxis
#         result[:, 0:3, 1] = yaxis
#         result[:, 0:3, 2] = zaxis
#         result[:, 3, 0] = -vec3.dot(xaxis, ori)
#         result[:, 3, 1] = -vec3.dot(yaxis, ori)
#         result[:, 3, 2] = -vec3.dot(zaxis, ori)
#         result[:, 3, 3] = 1.0
#         return result
#
#     @staticmethod
#     def backward(ctx: Any, *args: Any) -> Any:





class mat4(_GTensorBase):
    @staticmethod
    def inv_look_at(ori: vec3, dir: vec3, nor: vec3):
        dev = ori.device
        zaxis = dir
        xaxis = vec3.normalize(vec3.cross(nor, zaxis))
        yaxis = vec3.cross(zaxis, xaxis)
        exp_xaxis = torch.cat([xaxis, torch.zeros(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        exp_yaxis = torch.cat([yaxis, torch.zeros(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        exp_zaxis = torch.cat([zaxis, torch.zeros(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        exp_ori = torch.cat([ori, torch.ones(*xaxis.shape[:-1], 1).to(dev)], dim=-1).unsqueeze(-2)
        return mat4(torch.cat([exp_xaxis, exp_yaxis, exp_zaxis, exp_ori], dim=-2))

