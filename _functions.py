from ._internal import FunctionBase, device  #, ExtensionRendererModule
from .rendering import tensor, tensor_like, wrap
from typing import List, Tuple, Optional
import torch
from ._gmath import vec3
import os
import math


__FUNCTIONS_FOLDER__ = os.path.dirname(__file__).replace('\\', '/') + "/include/functions"

from .rendering.backend._vulkan_internal import syncronize_external_computation


def random_sphere_points(N, *, seed: int = 13, radius: float = 1.0):
    torch.manual_seed(seed)
    samples = torch.randn(N, 3)
    samples /= torch.sqrt((samples ** 2).sum(-1, keepdim=True))
    return vec3(samples * radius)


def random_equidistant_sphere_points(N, *, seed: int = 13, radius: float = 1.0):
    torch.manual_seed(103)
    initial_samples = torch.randn(N * 100, 3)
    initial_samples /= torch.sqrt((initial_samples ** 2).sum(-1, keepdim=True))
    initial_samples = initial_samples.numpy()
    from sklearn.cluster import KMeans
    import numpy as np
    kmeans = KMeans(n_clusters=N, random_state=seed).fit(initial_samples)
    CAMERAS = kmeans.cluster_centers_
    CAMERAS /= np.sqrt((CAMERAS ** 2).sum(-1, keepdims=True))
    CAMERAS *= radius
    return torch.as_tensor(CAMERAS, device=device())


def random_equidistant_camera_poses(N, *, seed: int = 13, radius: float = 1.0):
    origins = random_equidistant_sphere_points(N, seed=seed, radius=radius)
    camera_poses_tensor = torch.zeros(N, 9, device=device())
    camera_poses_tensor[:, 0:3] = origins
    camera_poses_tensor[:, 3:6] = vec3.normalize(-1*origins)
    camera_poses_tensor[:, 7] = 1.0
    return camera_poses_tensor


class _dummy_function(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/dummy_example/forward.comp.glsl',
        parameters = dict(
            a = torch.int64,
            b = torch.int64,
            out=torch.int64,
            alpha=float,
        )
    )

    def bind(self, a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0, out: Optional[torch.Tensor] = None) -> Tuple:
        self.alpha = alpha
        if out is None:
            out = tensor_like(a)
        self.a = wrap(a)
        self.b = wrap(b)
        self.out = wrap(out, 'out')
        return (a.numel(), 1, 1)

    def result(self) -> torch.Tensor:
        self.out.invalidate()
        return self.out.obj


def dummy_function(a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0, out: Optional[torch.Tensor] = None):
    return _dummy_function.eval(a, b, alpha = alpha, out = out)


class _random_ids(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/tools/random_ids.comp.glsl',
        parameters = dict(
            out_tensor = torch.int64,
            shape = [4, int],
            dim = int,
            output_dim = int,
        )
    )

    def bind(self, N: int, shape: Tuple[int], out: Optional[torch.Tensor] = None) -> Tuple:
        if out is None:
            out = tensor(N, len(shape), dtype=torch.long)
            # out = torch.zeros(N, len(shape), dtype=torch.long)
        self.out_tensor = wrap(out, 'out')
        self.dim = len(shape)
        for i in range(self.dim):
            self.shape[i] = shape[i]
        return (N, 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def random_ids(N: int, shape: Tuple[int], out: Optional[torch.Tensor] = None):
    return _random_ids.eval(N, shape, out = out)


class _gridtoimg(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/gridtoimg.comp.glsl",
        parameters = dict(
            in_tensor = torch.int64,
            out_tensor = torch.int64,
            shape = [4, int],
            dim = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> Tuple:
        out_shape = tuple(d - 1 for d in in_tensor.shape[:-1]) + (in_tensor.shape[-1],)
        if out is None:
            out = tensor(*out_shape, dtype=torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_tensor = wrap(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.shape[i] = in_tensor.shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


class _imgtogrid(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/imgtogrid.comp.glsl",
        parameters = dict(
            in_tensor = torch.int64,
            out_tensor = torch.int64,
            shape = [4, int],
            dim = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> Tuple:
        out_shape = tuple(d + 1 for d in in_tensor.shape[:-1]) + (in_tensor.shape[-1],)
        if out is None:
            out = tensor(*out_shape, dtype=torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_tensor = wrap(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.shape[i] = in_tensor.shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


class _Grid2ImageFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        in_tensor, = args
        return _gridtoimg.eval(in_tensor)

    @staticmethod
    def backward(ctx, *args):
        out_grad, = args
        return _imgtogrid.eval(out_grad)


class _Image2GridFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        in_tensor, = args
        return _imgtogrid.eval(in_tensor)

    @staticmethod
    def backward(ctx, *args):
        out_grad, = args
        return _gridtoimg.eval(out_grad)


def gridtoimg(in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is not None:
        assert not in_tensor.requires_grad
        return _gridtoimg.eval(in_tensor, out = out)
    return _Grid2ImageFunction.apply(in_tensor)


def imgtogrid(in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is not None:
        return _imgtogrid.eval(in_tensor, out = out)
    return _Image2GridFunction.apply(in_tensor)


class _resample_grid(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/resample_grid.comp.glsl",
        parameters=dict(
            in_tensor=torch.int64,
            out_tensor=torch.int64,
            in_shape=[4, int],
            out_shape=[4, int],
            dim=int,
            output_dim=int
        )
    )

    def bind(self, in_tensor: torch.Tensor, dst_shape: Tuple[int], out: Optional[torch.Tensor] = None) -> Tuple:
        assert len(in_tensor.shape) == len(dst_shape) + 1
        out_shape = dst_shape + (in_tensor.shape[-1],)
        if out is None:
            out = tensor(*out_shape, dtype=torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_tensor = wrap(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.in_shape[i] = in_tensor.shape[i]
            self.out_shape[i] = out_shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def resample_grid(in_tensor: torch.Tensor, dst_shape: Tuple[int,...], out: Optional[torch.Tensor] = None):
    dst_shape = tuple(dst_shape)
    min_shape = tuple((d + 1)//2 for d in in_tensor.shape[:-1])
    max_shape = tuple(d*2 for d in in_tensor.shape[:-1])
    clamp_shape = tuple(max(min(dst_shape[i], max_shape[i]), min_shape[i]) for i in range(len(dst_shape)))
    if clamp_shape == dst_shape:
        return _resample_grid.eval(in_tensor, dst_shape, out=out)
    g = _resample_grid.eval(in_tensor, clamp_shape, out=out)
    return resample_grid(g, dst_shape, out=out)


class _resample_img(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/resample_img.comp.glsl",
        parameters=dict(
            in_tensor=torch.int64,
            out_tensor=torch.int64,
            in_shape=[4, int],
            out_shape=[4, int],
            dim=int,
            output_dim=int
        )
    )

    def bind(self, in_tensor: torch.Tensor, dst_shape: Tuple[int], out: Optional[torch.Tensor] = None) -> Tuple:
        assert len(in_tensor.shape) == len(dst_shape) + 1
        out_shape = dst_shape + (in_tensor.shape[-1],)
        if out is None:
            out = tensor(*out_shape, dtype=torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_tensor = wrap(out, 'out')
        self.dim = len(out_shape) - 1
        self.output_dim = out_shape[-1]
        elements = 1
        for i in range(self.dim):
            self.in_shape[i] = in_tensor.shape[i]
            self.out_shape[i] = out_shape[i]
            elements *= out_shape[i]
        return (elements, 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def resample_img(in_tensor: torch.Tensor, dst_shape: Tuple[int,...], out: Optional[torch.Tensor] = None):
    dst_shape = tuple(dst_shape)
    min_shape = tuple((d + 1)//2 for d in in_tensor.shape[:-1])
    max_shape = tuple(d*2 for d in in_tensor.shape[:-1])
    clamp_shape = tuple(max(min(dst_shape[i], max_shape[i]), min_shape[i]) for i in range(len(dst_shape)))
    if clamp_shape == dst_shape:
        return _resample_img.eval(in_tensor, dst_shape, out=out)
    g = _resample_img.eval(in_tensor, clamp_shape, out=out)
    return resample_img(g, dst_shape, out=out)


def _power_of_two(x):
    return (x and (not (x & (x - 1))))


class _total_variation(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/regularizers/total_variation.comp.glsl',
        parameters= dict(
            in_tensor=torch.int64,
            out_tensor=torch.int64,
            shape=[4, int],
            dim=int
        )
    )

    def bind(self, in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> Tuple:
        assert len(in_tensor.shape) <= 4
        out_shape = in_tensor.shape[:-1] + (1,)
        if out is None:
            out = tensor(*out_shape, dtype=torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_tensor = wrap(out, 'out')
        self.dim = len(out_shape)-1
        for i in range(self.dim + 1):
            self.shape[i] = in_tensor.shape[i]
        return (math.prod(in_tensor.shape[:-1]), 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


class _total_variation_backward(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + '/regularizers/total_variation_backward.comp.glsl',
        parameters=dict(
            in_tensor=torch.int64,
            out_grad_tensor=torch.int64,
            in_grad_tensor=torch.int64,
            shape=[4, int],
            dim=int
        )
    )

    def bind(self, in_tensor: torch.Tensor, out_grad_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> Tuple:
        assert len(in_tensor.shape) <= 4
        out_shape = in_tensor.shape
        if out is None:
            out = torch.zeros(*out_shape, dtype=torch.float, device=device())
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_grad_tensor = wrap(out_grad_tensor, 'in')
        self.in_grad_tensor = wrap(out, 'out')
        self.dim = len(out_shape)-1
        for i in range(self.dim + 1):
            self.shape[i] = in_tensor.shape[i]
        return (math.prod(in_tensor.shape[:-1]), 1, 1)

    def result(self) -> torch.Tensor:
        self.in_grad_tensor.mark_as_dirty()
        self.in_grad_tensor.invalidate()
        return self.in_grad_tensor.obj


class _total_variation_diff(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        in_tensor, = args
        ctx.save_for_backward(in_tensor)
        return _total_variation.eval(in_tensor)

    @staticmethod
    def backward(ctx, *args):
        out_grad, = args
        in_tensor, = ctx.saved_tensors
        return _total_variation_backward.eval(in_tensor, out_grad)


def total_variation(in_tensor: torch.Tensor) -> torch.Tensor:
    return _total_variation_diff.apply(in_tensor)


class _copy_img_to_morton(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/copy_img_to_morton.comp.glsl",
        parameters = dict(
            in_tensor = torch.int64,
            out_tensor = torch.int64,
            resolution = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> Tuple:
        assert len(in_tensor.shape) == 3, 'in_tensor should be image (HxWxC)'
        assert in_tensor.shape[0] == in_tensor.shape[1], 'in_tensor should be square'
        resolution = in_tensor.shape[0]
        assert _power_of_two(resolution), 'in_tensor size should be power of two'
        out_shape = (resolution*resolution, in_tensor.shape[-1])
        if out is None:
            out = tensor(*out_shape, dtype=torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_tensor = wrap(out, 'out')
        self.resolution = resolution
        self.output_dim = out_shape[-1]
        return (resolution*resolution, 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def copy_img_to_morton(in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _copy_img_to_morton.eval(in_tensor, out = out)


class _copy_morton_to_img(FunctionBase):
    __extension_info__ = dict(
        path=__FUNCTIONS_FOLDER__ + "/tools/copy_morton_to_img.comp.glsl",
        parameters = dict(
            in_tensor = torch.int64,
            out_tensor = torch.int64,
            resolution = int,
            output_dim = int
        )
    )

    def bind(self, in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> Tuple:
        import math
        assert len(in_tensor.shape) == 2
        resolution = int(math.sqrt(in_tensor.shape[0]))
        assert in_tensor.shape[0] == resolution * resolution, 'linearization doesnt correspond to a square image'
        out_shape = (resolution, resolution, in_tensor.shape[-1])
        assert _power_of_two(resolution), 'resolution implicit in in_tensor should be power of two'
        if out is None:
            out = tensor(*out_shape, dtype=torch.float)
        else:
            assert out.shape == out_shape
        self.in_tensor = wrap(in_tensor, 'in')
        self.out_tensor = wrap(out, 'out')
        self.resolution = resolution
        self.output_dim = out_shape[-1]
        return (resolution*resolution, 1, 1)

    def result(self) -> torch.Tensor:
        self.out_tensor.mark_as_dirty()
        self.out_tensor.invalidate()
        return self.out_tensor.obj


def copy_morton_to_img(in_tensor: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _copy_morton_to_img.eval(in_tensor, out = out)


def create_density_quadtree(densities: torch.Tensor) -> torch.Tensor:
    resolution = densities.shape[0]
    assert len(densities.shape) == 3
    assert densities.shape[-1] == 1
    assert densities.shape[1] == resolution, 'Must be square'
    assert _power_of_two(resolution), "Resolution must be power of two"
    sizes = []
    size = resolution
    while size > 1:
        sizes.append(size * size)
        size //= 2
    sizes.reverse()
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)
    pdfs = torch.zeros(offsets[-1], 1, device=device())
    offsets = offsets[:-1]
    offsets.reverse()
    for o in offsets:
        copy_img_to_morton(densities, out=pdfs[o:o + densities.numel()])
        densities = resample_img(densities, (int(densities.shape[0]) // 2, int(densities.shape[1]) // 2)) * 4
    return pdfs


def model_to_tensor(model, shape: Tuple[int, int, int], bmin: vec3, bmax: vec3):
    dx = (bmax[0] - bmin[0]).item()/(shape[2] - 1)
    dy = (bmax[1] - bmin[1]).item()/(shape[1] - 1)
    dz = (bmax[2] - bmin[2]).item()/(shape[0] - 1)
    xs = torch.arange(bmin[0].item(), bmax[0].item() + 0.0000001, dx, device=device())
    ys = torch.arange(bmin[1].item(), bmax[1].item() + 0.0000001, dy, device=device())
    zs = torch.arange(bmin[2].item(), bmax[2].item() + 0.0000001, dz, device=device())
    points = torch.cartesian_prod(zs, ys, xs)[:, [2, 1, 0]]
    values = model(points)
    return values.view(*shape, -1)


class _DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """
    @staticmethod
    def forward(ctx, *args):
        input, min, max = args
        ctx.save_for_backward(input)
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, *args):
        grad_output, = args
        return grad_output.clone(), None, None


def dclamp(grid, min = 0.0, max = 1.0):
    return _DifferentiableClamp.apply(grid, min, max)


class EnhancingLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args):
        args = list(args)
        enhancing_process = args[-1]
        ctx.save_for_backward(*args[:-1])
        with torch.no_grad():
            output = enhancing_process(*args[:-1])
        return output

    @staticmethod
    def backward(ctx, *args):
        return tuple([*args])+(None,)  # None because the process is not differentiable


def enhance_output(*outputs, enhance_process):
    return EnhancingLayer.apply(*outputs, enhance_process)

