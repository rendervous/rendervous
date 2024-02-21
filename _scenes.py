import torch
import numpy as np
from . import device, constant, mat4, mat4x3, create_density_quadtree, resample_img, Image2D, ray_direction, mat3x4, \
    ray_to_segment
from ._maps import MapBase, parameter, ParameterDescriptor, Sampler, bind_parameter, RaycastableInfo, MeshInfo, \
    StructModule, map_struct, ParameterDescriptorLayoutType, TensorCheck
from ._internal import __INCLUDE_PATH__
from .rendering import vec3, structured_buffer, triangle_collection, aabb_collection, ads_model, instance_buffer, \
    ads_scene, \
    raytracing_manager, scratch_buffer, object_buffer, Buffer
from .rendering.backend._common import Layout, lazy_constant
from .rendering.backend._enums import MemoryLocation, BufferUsage
from typing import Optional, Union, List, Dict, Literal, Callable, Iterable
from .scenes import load_obj

#
# class Geometry:
#
#     def raycast_map(self):
#         '''
#         x, w -> t, surfel_code
#         '''
#         pass
#
#     def surfel_map(self):
#         '''
#         surfel_code -> P, N, etc. in world space
#         '''
#         pass
#
#
# class MeshGeometry:
#     pass


class Singleton(object):

    __instance__ = None

    @classmethod
    def get_instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = cls()
        return cls.__instance__


class Boundary(MapBase):
    __extension_info__ = None  # abstract node

    def __len__(self):
        raise NotImplementedError()


class GroupBoundary(Boundary):
    """
    Wraps a geometry to retrieve a distance map efficiently.
    Given a ray x,w returns t, is_entering and patch_index.
    """
    __extension_info__ = dict(
        parameters=dict(
            group_ads=torch.int64,
            patches=[-1, torch.int64]
        ),
        dynamics=[(6, 16)],
        generics=dict(
            INPUT_DIM=6, OUTPUT_DIM=3
        ),
        nodiff=True,
        code="""
    FORWARD
    {
        // input x, w,
        vec3 x = vec3(_input[0], _input[1], _input[2]);
        vec3 w = vec3(_input[3], _input[4], _input[5]);

        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery,              // Ray query
                            accelerationStructureEXT(parameters.group_ads),                  // Top-level acceleration structure
                            gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            x,                  // Ray origin
                            0.0,                   // Minimum t-value
                            w,            // Ray direction
                            10000.0);              // Maximum t-value

        // closest cached value among AABB candidates
        int patch_index = -1;
        float global_t = 100000.0;
        int from_outside = 0;

        while (rayQueryProceedEXT(rayQuery)) // traverse to find intersections
        {
            switch (rayQueryGetIntersectionTypeEXT(rayQuery, false))
            {
                case gl_RayQueryCandidateIntersectionTriangleEXT:
                    // any triangle intersection is accepted
                    rayQueryConfirmIntersectionEXT(rayQuery);
                    // rayQueryTerminateEXT(rayQuery);
                break;
                case gl_RayQueryCandidateIntersectionAABBEXT:
                    // any implicit hit is computed and cached in global surfel
                    // Get the instance (patch) hit
                    int index = 0;//rayQueryGetIntersectionInstanceIdEXT(rayQuery, false); // instance index
                    mat4x3 w2o = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, false);
                    vec3 tx = transform_position(x, w2o);
                    vec3 tw = transform_direction(w, w2o);
                    float _local_input[6] = float[6](tx.x, tx.y, tx.z, tw.x, tw.y, tw.z);
                    float _current[16];
                    RaycastableInfo raycastInfo = RaycastableInfo(parameters.patches[index]);
                    dynamic_forward(object, raycastInfo.callable_map, _local_input, _current);
                    float current_t;
                    int current_patch_index;
                    Surfel current_surfel;
                    if (hit2surfel(tx, tw, _current, current_t, current_patch_index, current_surfel) && (patch_index == -1 || current_t < global_t)) // replace closest surfel
                    {
                        patch_index = index;//current_patch_index + int_ptr(parameters.patch_offsets).data[index];
                        global_t = current_t;
                        from_outside = dot(tw, current_surfel.N) < 0 ? 1 : 0;
                    }
                    rayQueryGenerateIntersectionEXT(rayQuery, current_t);
                break;
            }
        }
        
        // check for the committed intersection to replace surfel if committed was a triangle
        if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
        {
            from_outside = rayQueryGetIntersectionFrontFaceEXT(rayQuery, true) ? 1 : 0;
            patch_index = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true); // instance index
            global_t = rayQueryGetIntersectionTEXT(rayQuery, true);
        }

        _output = float[3] (global_t, intBitsToFloat(from_outside), intBitsToFloat(patch_index));
    } 
        """
    )

    def __init__(self, geometry: 'GroupGeometry'):
        patches_info = geometry.per_patch_info()
        super().__init__(len(patches_info))
        self.geometry = geometry
        self.group_ads = geometry.group_ads
        for i, patch in enumerate(patches_info):
            self.patches[i] = patch

    def __len__(self):
        return len(self.geometry)


# GEOMETRIES

class Geometry(MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def signature():
        return (6, 16)

    # @staticmethod
    # def _build_corners_for_aabb(bmin, bmax):
    #     return torch.tensor([
    #         [bmin[0], bmin[1], bmin[2]],
    #         [bmin[0], bmin[1], bmax[2]],
    #         [bmin[0], bmax[1], bmin[2]],
    #         [bmin[0], bmax[1], bmax[2]],
    #         [bmax[0], bmin[1], bmin[2]],
    #         [bmax[0], bmin[1], bmax[2]],
    #         [bmax[0], bmax[1], bmin[2]],
    #         [bmax[0], bmax[1], bmax[2]],
    #     ], device=bmin.device
    #     )
    #
    # @staticmethod
    # def _transform_aabb(bmin, bmax, transform):
    #     corners = Geometry._build_corners_for_aabb(bmin, bmax)
    #     pts = (corners @ transform[:3, :3]) + transform[3:4, :]
    #     world_min, _ = pts.min(dim=0)
    #     world_max, _ = pts.max(dim=0)
    #     return world_min, world_max

    # def __init__(self, *args, **generics):
    #     super().__init__(*args, **generics)
    #     # self.object_bmin = torch.nn.Parameter(bmin, requires_grad=False)
    #     # self.object_bmax = torch.nn.Parameter(bmax, requires_grad=False)

    # def object_aabb(self):
    #     '''
    #     When implemented, returns bmin, bmax in object space for this geometry
    #     '''
    #     return self.object_bmin, self.object_bmax

    # def top_aabb(self):
    #     '''
    #     When implemented returns bmin, bmax after transform for this geometry
    #     '''
    #     corners = Geometry._build_corners_for_aabb(self.object_bmin, self.object_bmax)
    #     pts = (corners @ self._transform[:3, :3]) + self._transform[3:4, :]
    #     world_min, _ = pts.min(dim=0)
    #     world_max, _ = pts.max(dim=0)
    #     return world_min, world_max

    # def get_transform(self):
    #     return self._transform

    @staticmethod
    def _create_patch_buffer(callable: MapBase, mesh_info_buffer: Optional[Buffer] = None):
        buf = object_buffer(element_description=Layout.create_structure(
            mode='scalar',
            callable_map=torch.int64,
            mesh_info=torch.int64,
        )
        )
        with buf as b:
            b.callable_map = callable.__bindable__.device_ptr
            b.mesh_info = 0 if mesh_info_buffer is None else mesh_info_buffer.device_ptr
        return buf

    def __len__(self):
        '''
        When implemented, returns the number of patches for this geometry.
        '''
        return len(self.per_patch_info())

    def per_patch_info(self) -> List[int]:
        '''
        Gets the list of patch-descriptor references
        '''
        pass

    def per_patch_geometry(self) -> List[int]:
        '''
        Gets the list of patch geometry handles
        '''
        pass

    def update_patch_transforms(self, transforms: torch.Tensor):
        pass

    def update_ads_if_necessary(self) -> bool:
        pass

    def get_boundary(self) -> 'Boundary':
        raise NotImplementedError()  # TODO: IMPLEMENT A DEFAULT WRAPPER

    def _pre_eval(self, include_grads: bool = False):
        self.update_ads_if_necessary()
        return super()._pre_eval(include_grads)

    def transformed(self, T: torch.Tensor) -> 'TransformedGeometry':
        return TransformedGeometry(self, T)


class TransformedGeometry(Geometry):

    __extension_info__ = dict(
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=16),
        parameters=dict(
            base_geometry=MapBase,
            transform=ParameterDescriptor
        ),
        code="""
    FORWARD {
        vec3 x = vec3(_input[0], _input[1], _input[2]);
        vec3 w = vec3(_input[3], _input[4], _input[5]);
        mat4x3 T = mat4x3_ptr(parameters.transform.data).data[0];
        mat4x3 invT = inverse_transform(T);
        vec3 xt = transform_position(x, invT); 
        vec3 wt = transform_direction(w, invT);
        forward(parameters.base_geometry, float[6](xt.x, xt.y, xt.z, wt.x, wt.y, wt.z), _output);
        Surfel s;
        float t;
        int patch_index;
        hit2surfel(x, w, _output, t, patch_index, s);
        s = transform(s, T);
        surfel2array(t, patch_index, s, _output);
    }
    
    BACKWARD {
    }
        """
    )

    def __init__(self, base_geometry: Geometry, initial_transform: Optional[torch.Tensor] = None):
        super().__init__()
        self.base_geometry = base_geometry
        if initial_transform is None:
            initial_transform = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        self.transform = parameter(torch.nn.Parameter(initial_transform, requires_grad=True))
        self.transform_check = TensorCheck(initial_value=initial_transform)

    def per_patch_geometry(self) -> List[int]:
        return self.base_geometry.per_patch_geometry()

    def per_patch_info(self) -> List[int]:
        return self.base_geometry.per_patch_info()

    def update_ads_if_necessary(self) -> bool:
        return self.transform_check.changed(self.transform) | self.base_geometry.update_ads_if_necessary()

    def update_patch_transforms(self, transforms: torch.Tensor):
        with torch.no_grad():
            for t in transforms:
                t.copy_(mat3x4.composite(self.transform.T, t))
        self.base_geometry.update_patch_transforms(transforms)


class MeshGeometry(Geometry):
    __extension_info__ = dict(
        parameters=dict(
            mesh_ads=torch.int64,
            mesh_info=torch.int64,
        ),
        generics=dict(
            INPUT_DIM=6,  # x, w
            OUTPUT_DIM=16,  # t, N, G, C, T, B, patch_index
        ),
        nodiff=True,
        code="""
FORWARD
{
    // input x, w,
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    rayQueryEXT rayQuery_mesh;
    rayQueryInitializeEXT(rayQuery_mesh,              // Ray query
                        accelerationStructureEXT(parameters.mesh_ads),                  // Top-level acceleration structure
                        gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                        0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                        x,                  // Ray origin
                        0.0,                   // Minimum t-value
                        w,            // Ray direction
                        10000.0);              // Maximum t-value

    while(rayQueryProceedEXT(rayQuery_mesh))
        rayQueryConfirmIntersectionEXT(rayQuery_mesh);

    if (rayQueryGetIntersectionTypeEXT(rayQuery_mesh, true) != gl_RayQueryCommittedIntersectionNoneEXT)
    {
        int index = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery_mesh, true);
        vec2 bar = rayQueryGetIntersectionBarycentricsEXT(rayQuery_mesh, true); 
        float t = rayQueryGetIntersectionTEXT(rayQuery_mesh, true);
        surfel2array(t, 0, sample_surfel(MeshInfo(parameters.mesh_info), index, bar), _output);
    }
    else
        noHit2array(_output);
}
        """
    )

    @staticmethod
    def compute_normals(positions: torch.Tensor, indices: torch.Tensor):
        normals = torch.zeros_like(positions)
        indices = indices.long()  # to be used
        P0 = positions[indices[:, 0]]
        P1 = positions[indices[:, 1]]
        P2 = positions[indices[:, 2]]
        V1 = vec3.normalize(P1 - P0)
        V2 = vec3.normalize(P2 - P0)
        N = torch.cross(V1, V2)  # do not normalize for proper weight
        indices0 = indices[:, 0].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices0, N)
        indices1 = indices[:, 1].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices1, N)
        indices2 = indices[:, 2].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices2, N)
        return normals / torch.sqrt((normals ** 2).sum(-1, keepdim=True))

    @staticmethod
    def load_obj(path: str) -> 'MeshGeometry':
        obj = load_obj(path)
        pos = obj['buffers']['P']
        bmin, _ = pos.min(dim=0)
        bmax, _ = pos.max(dim=0)
        pos = (pos - bmin - (bmax - bmin) * 0.5) * 2 / (bmax - bmin).max()
        return MeshGeometry(positions=pos, indices=obj['buffers']['P_indices'])

    @staticmethod
    def box():
        return MeshGeometry(
            positions=torch.tensor(
                [
                    # Neg z
                    [-0.5, -0.5, -0.5],  # 0
                    [0.5, -0.5, -0.5],  # 1
                    [-0.5, 0.5, -0.5],  # 2
                    [0.5, 0.5, -0.5],  # 3
                    # Pos z

                    [-0.5, -0.5, 0.5],  # 4
                    [0.5, -0.5, 0.5],  # 5
                    [-0.5, 0.5, 0.5],  # 6
                    [0.5, 0.5, 0.5],  # 7

                    [-0.5, -0.5, -0.5],  # 8
                    [0.5, -0.5, -0.5],  # 9
                    [-0.5, -0.5, 0.5],  # 10
                    [0.5, -0.5, 0.5],  # 11

                    [-0.5, 0.5, -0.5],  # 12
                    [0.5, 0.5, -0.5],  # 13
                    [-0.5, 0.5, 0.5],  # 14
                    [0.5, 0.5, 0.5],  # 15

                    [-0.5, -0.5, -0.5],  # 16
                    [-0.5, 0.5, -0.5],  # 17
                    [-0.5, -0.5, 0.5],  # 18
                    [-0.5, 0.5, 0.5],  # 19

                    [0.5, -0.5, -0.5],  # 20
                    [0.5, 0.5, -0.5],  # 21
                    [0.5, -0.5, 0.5],  # 22
                    [0.5, 0.5, 0.5],  # 23
                ],
                device=device()
            ),
            # positions=torch.tensor(
            #     [
            #         # Neg z
            #         [-1.0, -1.0, -1.0],  # 0
            #         [1.0, -1.0, -1.0],  # 1
            #         [-1.0, 1.0, -1.0],  # 2
            #         [1.0, 1.0, -1.0],  # 3
            #         # Pos z
            #
            #         [-1.0, -1.0, 1.0],  # 4
            #         [1.0, -1.0, 1.0],  # 5
            #         [-1.0, 1.0, 1.0],  # 6
            #         [1.0, 1.0, 1.0],  # 7
            #
            #         [-1.0, -1.0, -1.0],  # 8
            #         [1.0, -1.0, -1.0],  # 9
            #         [-1.0, -1.0, 1.0],  # 10
            #         [1.0, -1.0, 1.0],  # 11
            #
            #         [-1.0, 1.0, -1.0],  # 12
            #         [1.0, 1.0, -1.0],  # 13
            #         [-1.0, 1.0, 1.0],  # 14
            #         [1.0, 1.0, 1.0],  # 15
            #
            #         [-1.0, -1.0, -1.0],  # 16
            #         [-1.0, 1.0, -1.0],  # 17
            #         [-1.0, -1.0, 1.0],  # 18
            #         [-1.0, 1.0, 1.0],  # 19
            #
            #         [1.0, -1.0, -1.0],  # 20
            #         [1.0, 1.0, -1.0],  # 21
            #         [1.0, -1.0, 1.0],  # 22
            #         [1.0, 1.0, 1.0],  # 23
            #     ],
            #     device=device()
            # ),
            indices=torch.tensor(
                [
                    [0, 2, 3],
                    [0, 3, 1],
                    [4, 7, 6],
                    [4, 5, 7],

                    [8, 11, 10],
                    [8, 9, 11],
                    [12, 14, 15],
                    [12, 15, 13],

                    [16, 18, 19],
                    [16, 19, 17],
                    [20, 23, 22],
                    [20, 21, 23],
                ], dtype=torch.int32, device=device()
            ), normals=None, uvs=torch.tensor([
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ], device=device()),
            normalize=True
        )

    def __init__(self, positions: torch.Tensor, normals: Optional[torch.Tensor] = None,
                 uvs: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None, normalize: bool = False, compute_normals: bool = True):
        # if indices is None:  # assume default indices for all positions
        #     assert len(positions) % 3 == 0, 'If no indices is provided, positions should be divisible by 3'
        #     indices = torch.arange(0, len(positions), 1, dtype=torch.int32, device=positions.device).view(-1, 3)
        if normals is None and compute_normals:
            normals = MeshGeometry.compute_normals(positions, indices)
        super().__init__()
        if normalize:
            with torch.no_grad():
                bmin, _ = positions.min(dim=0)
                bmax, _ = positions.max(dim=0)
                positions = (positions - bmin - (bmax - bmin) * 0.5) * 2 / (bmax - bmin).max()
        self.mesh_info_buffer = object_buffer(element_description=Layout.create_structure(
            mode='scalar',
            positions=ParameterDescriptorLayoutType,
            normals=ParameterDescriptorLayoutType,
            coordinates=ParameterDescriptorLayoutType,
            tangents=ParameterDescriptorLayoutType,
            binormals=ParameterDescriptorLayoutType,
            indices=ParameterDescriptorLayoutType
        ), memory=MemoryLocation.GPU, usage=BufferUsage.STORAGE)
        self.mesh_info_module = StructModule(self.mesh_info_buffer.accessor)
        self.mesh_info_module.positions = parameter(positions)
        self.mesh_info_module.normals = parameter(normals)
        self.mesh_info_module.coordinates = parameter(uvs)
        self.mesh_info_module.tangents = parameter(None)
        self.mesh_info_module.binormals = parameter(None)
        self.mesh_info_module.indices = parameter(indices)
        self.mesh_info = self.mesh_info_buffer.device_ptr
        self._patch_info = Geometry._create_patch_buffer(self, self.mesh_info_buffer)
        self._per_patch_info = [self._patch_info.device_ptr]
        # create bottom ads
        # Create vertices for the triangle
        vertices = structured_buffer(
            count=len(positions),
            element_description=dict(
                position=vec3
            ),
            usage=BufferUsage.RAYTRACING_RESOURCE,
            memory=MemoryLocation.GPU
        )
        # Create indices for the faces
        indices_buffer = structured_buffer(
            count=len(indices) * 3,
            element_description=int,
            usage=BufferUsage.RAYTRACING_RESOURCE,
            memory=MemoryLocation.GPU
        )
        # vertices.load([ vec3(-0.6, 0, 0), vec3(0.6, 0, 0), vec3(0, 1, 0)])
        with vertices.map('in') as v:
            # v.position refers to all positions at once...
            v.position = positions
        indices_buffer.load(indices.view(-1))
        # Create a triangle collection
        self.geometry = triangle_collection()
        self.geometry.append(vertices=vertices, indices=indices_buffer)
        # Create a bottom ads with the geometry
        self.geometry_ads = ads_model(self.geometry)

        self._per_patch_geometry = [self.geometry_ads.handle]

        # Create an instance buffer for the top level objects
        self.scene_buffer = instance_buffer(1, MemoryLocation.GPU)
        with self.scene_buffer.map('in', clear=True) as s:
            s.flags = 0
            s.mask = 0xFF
            # By default, all other values of the instance are filled
            # for instance, transform with identity transform and 0 offset.
            # mask with 255
            s.transform[0][0] = 1.0
            s.transform[1][1] = 1.0
            s.transform[2][2] = 1.0
            s.accelerationStructureReference = self.geometry_ads.handle

        # Create the top level ads
        self.scene_ads = ads_scene(self.scene_buffer)

        # scratch buffer used to build the ads shared for all ads'
        self.scratch_buffer = scratch_buffer(self.geometry_ads, self.scene_ads)
        self.position_checker = TensorCheck()
        self.mesh_ads = self.scene_ads.handle
        self.update_ads_if_necessary()

    def _pre_eval(self, include_grads: bool = False):
        super()._pre_eval(include_grads)
        self.mesh_info_buffer.update_gpu()

    def update_ads_if_necessary(self) -> bool:
        if self.position_checker.changed(self.mesh_info_module.positions):
            with raytracing_manager() as man:
                man.build_ads(self.geometry_ads, self.scratch_buffer)
                man.build_ads(self.scene_ads, self.scratch_buffer)
            return True
        return False

    def __len__(self):
        return 1

    def per_patch_geometry(self) -> List[int]:
        return self._per_patch_geometry

    def per_patch_info(self) -> List[int]:
        return self._per_patch_info

    def update_patch_transforms(self, transforms: torch.Tensor):
        return


class GroupGeometry(Geometry):
    __extension_info__ = dict(
        dynamics=[(6, 16)],  # can call dynamically to other geometries
        parameters=dict(
            group_ads=torch.int64,
            patches=[-1, torch.int64]  # flatten representation of all patches
        ),
        generics=dict(
            INPUT_DIM=6,  # x, w
            OUTPUT_DIM=16,  # t, N, G, C, T, B, patch_index
        ),
        nodiff=True,
        code="""
    FORWARD
    {
        // input x, w,
        vec3 x = vec3(_input[0], _input[1], _input[2]);
        vec3 w = vec3(_input[3], _input[4], _input[5]);

        rayQueryEXT rayQuery;
        rayQueryInitializeEXT(rayQuery,              // Ray query
                            accelerationStructureEXT(parameters.group_ads),                  // Top-level acceleration structure
                            gl_RayFlagsOpaqueEXT,  // Ray flags, here saying "treat all geometry as opaque"
                            0xFF,                  // 8-bit instance mask, here saying "trace against all instances"
                            x,                  // Ray origin
                            0.0,                   // Minimum t-value
                            w,            // Ray direction
                            10000.0);              // Maximum t-value

        // closest cached value among AABB candidates
        int patch_index = -1;
        float global_t = 100000.0;
        Surfel global_surfel = Surfel(vec3(0.0), vec3(0.0), vec3(0.0), vec2(0.0), vec3(0.0), vec3(0.0));

        while (rayQueryProceedEXT(rayQuery)) // traverse to find intersections
        {
            switch (rayQueryGetIntersectionTypeEXT(rayQuery, false))
            {
                case gl_RayQueryCandidateIntersectionTriangleEXT:
                    // any triangle intersection is accepted
                    rayQueryConfirmIntersectionEXT(rayQuery);
                    //rayQueryTerminateEXT(rayQuery);
                break;
                case gl_RayQueryCandidateIntersectionAABBEXT:
                    // any implicit hit is computed and cached in global surfel

                    // Get the instance (patch) hit
                    int index = 0;//rayQueryGetIntersectionInstanceIdEXT(rayQuery, false); // instance index
                    mat4x3 w2o = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, false);
                    mat4x3 o2w = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, false);

                    vec3 tx = transform_position(x, w2o);
                    vec3 tw = transform_direction(w, w2o);
                    float _local_input[6] = float[6](tx.x, tx.y, tx.z, tw.x, tw.y, tw.z);
                    float _current[16];
                    RaycastableInfo raycastInfo = RaycastableInfo(parameters.patches[index]);
                    dynamic_forward(object, raycastInfo.callable_map, _local_input, _current);
                    float current_t;
                    int current_patch_index;
                    Surfel current_surfel;
                    if (hit2surfel(tx, tw, _current, current_t, current_patch_index, current_surfel) && (patch_index == -1 || current_t < global_t)) // replace closest surfel
                    {
                        patch_index = index;//current_patch_index + int_ptr(parameters.patch_offsets).data[index];
                        global_t = current_t;
                        global_surfel = transform(current_surfel, o2w);
                    }
                    rayQueryGenerateIntersectionEXT(rayQuery, current_t);
                break;
            }
        }
        
        // check for the committed intersection to replace surfel if committed was a triangle
        if (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
        {
            mat4x3 o2w = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
            int index = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true); // instance index
            int primitive_index = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
            vec2 baricentrics = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
            global_t = rayQueryGetIntersectionTEXT(rayQuery, true);
            RaycastableInfo raycastInfo = RaycastableInfo(parameters.patches[index]);
            MeshInfo meshInfo = MeshInfo(raycastInfo.mesh_info);
            global_surfel = transform(sample_surfel(meshInfo, primitive_index, baricentrics), o2w);        
            patch_index = index;
        }

        surfel2array(global_t, patch_index, global_surfel, _output);
    }
            """
    )

    __GEOMETRY_ID__ = 0

    @staticmethod
    def _flatten_patches(geometries: Iterable[Geometry]):
        patch_geometries = []
        patch_infos = []
        for g in geometries:
            patch_infos.extend(g.per_patch_info())
            patch_geometries.extend(g.per_patch_geometry())
        return patch_infos, patch_geometries

    def __init__(self, *geometries: Geometry):
        GroupGeometry.__GEOMETRY_ID__ += 1
        # Tag with a generic that a new geometry collection is needed to differentiate from others within nested dynamic calls
        _per_patch_infos, _per_patch_geometries = GroupGeometry._flatten_patches(geometries)
        super().__init__(len(_per_patch_infos), RDV_GEOMETRY_ID=GroupGeometry.__GEOMETRY_ID__)
        self._per_patch_infos, self._per_patch_geometries = _per_patch_infos, _per_patch_geometries
        self._per_patch_geometries_tensor = torch.tensor(_per_patch_geometries, dtype=torch.int64).unsqueeze(-1)
        self.geometries_module_list = torch.nn.ModuleList(geometries)
        self.geometries = geometries
        for i, patch in enumerate(self._per_patch_infos):
            self.patches[i] = patch
        # Create an instance buffer for the top level objects
        self.scene_buffer = instance_buffer(len(self._per_patch_geometries), MemoryLocation.GPU)
        # Create the top level ads
        self.scene_ads = ads_scene(self.scene_buffer)
        # scratch buffer used to build the ads shared for all ads'
        self.scratch_buffer = scratch_buffer(self.scene_ads)
        self.group_ads = self.scene_ads.handle
        self.cached_transforms = torch.empty(len(self._per_patch_geometries), 3, 4)
        self.cached_transforms[:] = GroupGeometry.__identity__
        self.update_ads_if_necessary()

    __identity__ = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    def per_patch_geometry(self) -> List[int]:
        return self._per_patch_geometries

    def per_patch_info(self) -> List[int]:
        return self._per_patch_infos

    def update_patch_transforms(self, transforms: torch.Tensor):
        offset = 0
        for g in self.geometries:
            p = len(g)
            g.update_patch_transforms(transforms[offset:offset + p])
            offset += p

    def update_ads_if_necessary(self):
        any_geometry_changed = False
        for g in self.geometries:
            any_geometry_changed |= g.update_ads_if_necessary()
        current_transforms = torch.empty(len(self._per_patch_geometries), 3, 4)
        current_transforms[:] = GroupGeometry.__identity__
        self.update_patch_transforms(current_transforms)
        if any_geometry_changed or not torch.equal(current_transforms, self.cached_transforms):
            with self.scene_buffer.map('in') as s:
                # notice that these updates are serial
                s.flags = 0
                s.mask = 0xFF
                s.accelerationStructureReference = self._per_patch_geometries_tensor
                s.transform = current_transforms
            with raytracing_manager() as man:
                man.build_ads(self.scene_ads, self.scratch_buffer)
            self.cached_transforms = current_transforms


    def get_boundary(self) -> 'Boundary':
        return GroupBoundary(self)


# ENVIRONMENTS


class Environment(MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: Optional[str] = None,
            parameters: Optional[Dict] = None,
            external_code: Optional[str] = None
    ):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        if external_code is None:
            external_code = ''
        return dict(
            generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
            parameters=parameters,
            code=f"""

        {external_code}

        FORWARD {{
            vec3 x = vec3(_input[0], _input[1], _input[2]);
            vec3 w = vec3(_input[3], _input[4], _input[5]);
            vec3 E;
            {fw_code}
            _output = float[3](E.x, E.y, E.z);
        }}

        BACKWARD {{
            {bw_code}
        }}
                """
        )


class EnvironmentSamplerPDF(MapBase):
    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: Optional[str] = None,
            parameters: Optional[Dict] = None,
            external_code: Optional[str] = None
    ):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        if external_code is None:
            external_code = ''
        return dict(
            generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
            parameters=parameters,
            code=f"""

        {external_code}

        FORWARD {{
            vec3 x = vec3(_input[0], _input[1], _input[2]);
            vec3 w = vec3(_input[3], _input[4], _input[5]);
            float pdf;
            {fw_code}
            _output = float[1](pdf);
        }}

        BACKWARD {{
            {bw_code}
        }}
                """
        )


class EnvironmentSampler(MapBase):  # x -> we, E(we)/pdf(we), pdf(we)

    __extension_info__ = None  # abstract node

    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: Optional[str] = None,
            parameters: Optional[Dict] = None,
            external_code: Optional[str] = None
    ):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        if external_code is None:
            external_code = ''
        return dict(
            generics = dict(INPUT_DIM=3, OUTPUT_DIM=7),
            parameters = parameters,
            code = f"""
            
    {external_code}
    
    FORWARD {{
        vec3 x = vec3(_input[0], _input[1], _input[2]);
        vec3 w;
        vec3 E;
        float pdf;
        {fw_code}
        _output = float[7](w.x, w.y, w.z, E.x, E.y, E.z, pdf);
    }}
    
    BACKWARD {{
        {bw_code}
    }}
            """
        )

    def get_environment(self) -> Environment:  # x, we -> E(we)
        raise NotImplementedError()

    def get_pdf(self) -> EnvironmentSamplerPDF:   # x, we -> pdf(we)
        raise NotImplementedError()


class XREnvironment(Environment):
    __extension_info__ = Environment.create_extension_info(
        parameters=dict(
            environment_img=MapBase
        ),
        fw_code=f"""
        vec2 xr = dir2xr(w);
        float[3] R;
        forward(parameters.environment_img, float[2](xr.x, xr.y), R);
        E = vec3(R[0], R[1], R[2]); 
        """
    )
    def __init__(self, environment_img: MapBase):
        super().__init__()
        self.environment_img = environment_img


class XREnvironmentSamplerPDF(EnvironmentSamplerPDF):
    __extension_info__ = EnvironmentSamplerPDF.create_extension_info(
        parameters=dict(
            densities=torch.Tensor,  # quadtree probabilities
            levels=int,  # Number of levels of the quadtree
        ),
        fw_code="""
        // compute w converted to pixel px, py and to z-order index
        vec2 xr = dir2xr(w);
        vec2 c = xr * 0.5 + vec2(0.5); // 0,0 - 1,1
        ivec2 p = ivec2(clamp(c, vec2(0.0), vec2(0.999999)) * (1 << parameters.levels));
        int index = pixel2morton(p);
        // compute last level offset 4*(4^(levels - 1) - 1)/3 and peek pdf of the cell 
        pdf = 1.0;
        if (parameters.levels > 0) // more than one node
        {
            int offset = 4 * (1 << (2 * (parameters.levels - 1)) - 1) / 3;
            // peek the density and compute final point pdf
            float_ptr densities_buf = float_ptr(parameters.densities);
            pdf = densities_buf.data[offset + index];
        }
        // multiply by peek in that area
        vec2 p0 = vec2(p) / (1 << parameters.levels);
        vec2 p1 = vec2(p + ivec2(1)) / (1 << parameters.levels);
        float pixel_area = 2 * pi * (cos(p0.y*pi) - cos(p1.y*pi)) / (1 << parameters.levels);
        pdf *= 1.0 / pixel_area;
        """
    )

    def __init__(self, densities: torch.Tensor, levels: int):
        super().__init__()
        self.densities = densities
        self.levels = levels


class XREnvironmentSampler(EnvironmentSampler):
    __extension_info__ = EnvironmentSampler.create_extension_info(
        parameters=dict(
            environment=MapBase,  # Image with radiances
            densities=torch.Tensor,  # quadtree probabilities
            levels=int,  # Number of levels of the quadtree
        ),
        external_code="""
    #include "sc_environment.h"
        """,
        fw_code="""
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
        w = randomDirection((p0.x * 2 - 1) * pi, (p1.x * 2 - 1) * pi, p0.y * pi, p1.y * pi);
        environment(object, x, w, E);
        E *= pixel_area / max(0.000000001, prob);
        pdf = max(0.000000001, prob) / pixel_area;
        """)

    def __init__(self, environment_img: MapBase, quadtree: torch.Tensor, levels: int):
        super().__init__()
        self.environment_img = environment_img
        self.environment = XREnvironment(environment_img)
        self.densities = quadtree
        self.levels = levels
        self.pdf = XREnvironmentSamplerPDF(self.densities, self.levels)

    def get_environment(self) -> Environment:
        return self.environment

    def get_pdf(self) -> EnvironmentSamplerPDF:
        return self.pdf

    @staticmethod
    def from_tensor(environment_tensor: torch.Tensor, levels: int = 10) -> 'XREnvironmentSampler':
        resolution = 1 << levels
        densities = resample_img(environment_tensor.sum(-1, keepdim=True), (resolution, resolution))
        for py in range(resolution):
            w = np.cos(py * np.pi / resolution) - np.cos((py + 1) * np.pi / resolution)
            densities[py, :, :] *= w
        densities /= max(densities.sum(), 0.00000001)
        quadtree = create_density_quadtree(densities)
        return XREnvironmentSampler(Image2D(environment_tensor), quadtree, levels)


# class UniformEnvironmentSampler(EnvironmentSampler):
#     __extension_info__ = EnvironmentSampler.create_extension_info("""
#         vec3 we = randomDirection();
#         pdf = 1/(4 * pi);
#         E = environment(object, x, we);
#     """, parameters=dict(
#         environment=MapBase
#     ))


# SURFACES


class SurfaceScattering(MapBase):  # wi, wo, P, N, G, C, T, B-> W
    """
    Represents a cosine weighted BSDF
    """
    __extension_info__ = None  # Abstract Node
    @staticmethod
    def create_extension_info(
        fw_code: str,
        bw_code: Optional[str] = None,
        external_code: Optional[str] = None,
        parameters: Optional[Dict] = None
    ):
        if parameters is None:
            parameters = {}
        if external_code is None:
            external_code = ''
        if bw_code is None:
            bw_code = f"""
            """
        return dict(
            generics=dict(INPUT_DIM=23, OUTPUT_DIM=3),
            parameters=parameters,
            code=f"""
{external_code}

FORWARD {{
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 wout = vec3(_input[3], _input[4], _input[5]);
    Surfel surfel = Surfel(
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec3(_input[12], _input[13], _input[14]),
        vec2(_input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]),
        vec3(_input[20], _input[21], _input[22]));
    bool from_outside = dot(win, surfel.G) < 0;
    bool correct_hemisphere = (dot(win, surfel.N) < 0) == from_outside;
    vec3 N = correct_hemisphere ? surfel.N : surfel.G; 
    vec3 fN = from_outside ? N : -N;
    bool is_transmission = dot(wout, fN) < 0;
    vec3 W;
    {fw_code}
    _output = float[3](W.x, W.y, W.z);
}}

BACKWARD {{
    {bw_code}
}}
            """
        )

    @staticmethod
    def signature():
        return (23, 3)


class SurfaceScatteringSamplerPDF(MapBase):  # wi, wo, P, N, G, C, T, B-> pdf(wo)
    """
    Represents the outgoing direction pdf of a cosine weighted BSDF sampler
    """
    __extension_info__ = None  # Abstract Node
    @staticmethod
    def create_extension_info(
        fw_code: str,
        bw_code: Optional[str] = None,
        external_code: Optional[str] = None,
        parameters: Optional[Dict] = None
    ):
        if parameters is None:
            parameters = {}
        if external_code is None:
            external_code = ''
        if bw_code is None:
            bw_code = f"""
            """
        return dict(
            generics=dict(INPUT_DIM=23, OUTPUT_DIM=1),
            parameters=parameters,
            code=f"""
{external_code}

FORWARD {{
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 wout = vec3(_input[3], _input[4], _input[5]);
    Surfel surfel = Surfel(
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec3(_input[12], _input[13], _input[14]),
        vec2(_input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]),
        vec3(_input[20], _input[21], _input[22]));
    bool from_outside = dot(win, surfel.G) < 0;
    bool correct_hemisphere = (dot(win, surfel.N) < 0) == from_outside;
    vec3 N = correct_hemisphere ? surfel.N : surfel.G; 
    vec3 fN = from_outside ? N : -N;
    bool is_transmission = dot(wout, fN) < 0;
    float pdf;
    {fw_code}
    _output[0] = pdf;
}}

BACKWARD {{
    {bw_code}
}}
            """
        )

    @staticmethod
    def signature():
        return (23, 1)


class SurfaceScatteringSampler(MapBase):  # win, N, C, P, T, B -> wout, W/pdf(wo), pdf(W,wo)
    __extension_info__ = None  # Abstract Node
    @staticmethod
    def create_extension_info(
        fw_code: str,
        bw_code: Optional[str] = None,
        external_code: Optional[str] = None,
        parameters: Optional[Dict] = None
    ):
        if parameters is None:
            parameters = {}
        if external_code is None:
            external_code = ''
        if bw_code is None:
            bw_code = ''
        return dict(
            generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
            parameters=parameters,
            code=f"""
{external_code}

FORWARD {{
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    Surfel surfel = Surfel(
        vec3(_input[3], _input[4], _input[5]),
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec2(_input[12], _input[13]),
        vec3(_input[14], _input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]));
    bool from_outside = dot(win, surfel.G) < 0;
    bool correct_hemisphere = (dot(win, surfel.N) < 0) == from_outside;
    vec3 N = correct_hemisphere ? surfel.N : surfel.G; 
    vec3 fN = from_outside ? N : -N;
    vec3 wout = vec3(0.0);
    vec3 W = vec3(0.0);
    float pdf = 0.0;
    {fw_code}
    _output = float[7](wout.x, wout.y, wout.z, W.x, W.y, W.z, pdf);
}}

BACKWARD {{
    {bw_code}
}}
            """
        )

    @staticmethod
    def signature():
        return (20, 7)

    def get_scattering(self):
        raise NotImplementedError()

    def get_pdf(self):
        raise NotImplementedError()


class LambertSurfaceScattering(SurfaceScattering, Singleton):
    __extension_info__ = SurfaceScattering.create_extension_info(
        f"""
    float cosine_theta = dot(wout, fN);
    W = vec3(is_transmission ? 0.0 : cosine_theta / pi);
        """
    )


class LambertSurfaceScatteringSamplerPDF(SurfaceScatteringSamplerPDF, Singleton):
    __extension_info__ = SurfaceScatteringSamplerPDF.create_extension_info(f"""
    float cosine_theta = dot(wout, fN);
    pdf = is_transmission ? 0.0 : cosine_theta / pi;
    """)


class LambertSurfaceScatteringSampler(SurfaceScatteringSampler, Singleton):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(
        f"""
    float cosine_theta;
    if (!from_outside)
    {{
        wout = vec3(0.0);
        pdf = 0.0;
        W = vec3(0.0);
    }}
    else {{
        wout = randomHSDirectionCosineWeighted(fN, cosine_theta);
        pdf = cosine_theta / pi;
        W = vec3(1.0);
    }}
        """
    )

    def get_scattering(self):
        return LambertSurfaceScattering.get_instance()

    def get_pdf(self):
        return LambertSurfaceScatteringSamplerPDF.get_instance()


class DeltaSurfaceScattering(SurfaceScattering, Singleton):
    __extension_info__ = SurfaceScattering.create_extension_info(f"""
    W = vec3(0.0);
    """)


class DeltaSurfaceScatteringPDF(SurfaceScatteringSamplerPDF, Singleton):
    __extension_info__ = SurfaceScatteringSamplerPDF.create_extension_info(f"""
    pdf = 0.0;
    """)


class FresnelSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(f"""
    vec3 reflect_w = reflect(win, fN);
    float eta = from_outside ? 1 / parameters.refraction_index : parameters.refraction_index;
    vec3 refract_w = refract(win, fN, eta);
    
    float theta = abs(dot(win, fN));
    float f = pow((1 - eta)/(1 + eta), 2);
    float R = f + (1 - f) * pow(1 - theta, 5);
    
    if (refract_w == vec3(0.0) || random() < R) {{
        wout = reflect_w;
    }}
    else {{
        wout = refract_w;
    }}
    W = vec3(1.0/(eta*eta));
    pdf = -1.0;
    """,
    parameters=dict(
        refraction_index=float
    )
    )

    def __init__(self, refraction_index: float = 1.5):
        super().__init__()
        self.refraction_index = refraction_index

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class MirrorSurfaceScatteringSampler(SurfaceScatteringSampler, Singleton):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(f"""
    wout = reflect(win, fN);
    W = vec3(1.0);
    pdf = -1.0;
    """)

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class SpectralFresnelSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = SurfaceScatteringSampler.create_extension_info(f"""
    vec3 reflect_w = reflect(win, fN);
    float spectral_alpha = random() * 6;
    int index = int(spectral_alpha);
    float refraction_index = parameters.refraction_index*(1 + 0.01*spectral_alpha);
    vec3 colors [6] = {{ 
        vec3(3.0, 0.0, 0.0), 
        vec3(1.5, 1.5, 0.0),
        vec3(0.0, 3.0, 0.0),
        vec3(0.0, 1.5, 1.5),
        vec3(0.0, 0.0, 3.0), 
        vec3(1.5, 0.0, 1.5)
    }};
    vec3 Cr = colors[index];
    float eta = from_outside ? 1/refraction_index : refraction_index;
    vec3 refract_w = refract(win, fN, eta);

    float theta = -dot(win, fN);
    float f = pow((1 - 1/refraction_index)/(1 + 1/refraction_index), 2);
    float R = f + (1 - f) * pow(1 - theta, 5);

    if (refract_w == vec3(0.0) || random() < R) {{
        wout = reflect_w;
    }}
    else {{
        wout = refract_w;
    }}
    W = Cr / vec3(eta*eta);
    
    """,
                                                                        parameters=dict(
                                                                            refraction_index=float
                                                                        )
                                                                        )

    def __init__(self, refraction_index: float = 1.5):
        super().__init__()
        self.refraction_index = refraction_index

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class AlbedoSurfaceScattering(SurfaceScattering):
    __extension_info__ = dict(
        parameters=dict(
            base_scattering=MapBase,
            albedo=MapBase,
        ),
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=3),
        code=f"""
    FORWARD {{
        forward(parameters.base_scattering, _input, _output);
        float[3] albedo;
    #if ALBEDO_MAP_DIM == 2
        float[2] C = float[2](_input[15],_input[16]);
        forward(parameters.albedo, C, albedo);
    #endif
    #if ALBEDO_MAP_DIM == 3
        float[3] P = float[3](_input[6], _input[7], _input[8]);
        forward(parameters.albedo, P, albedo);
    #endif
        _output[0] *= albedo[0];
        _output[1] *= albedo[1];
        _output[2] *= albedo[2];
    }}
        """,
        nodiff=True
    )

    def __init__(self, base: SurfaceScattering, albedo: MapBase):
        assert albedo.input_dim == 2 or albedo.input_dim == 3, 'Can not use a map to albedo that is not 2D or 3D'
        assert albedo.output_dim == 3
        super().__init__(ALBEDO_MAP_DIM = albedo.input_dim)
        self.base_scattering = base
        self.albedo = albedo


class AlbedoSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = dict(
        parameters=dict(
            base_scattering_sampler=MapBase,
            albedo=MapBase
        ),
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        nodiff=True,
        code=f"""
    FORWARD {{
        forward(parameters.base_scattering_sampler, _input, _output);
        float[3] albedo;
        #if ALBEDO_MAP_DIM == 2
        float[2] C = float[2](_input[12],_input[13]);
        forward(parameters.albedo, C, albedo);
    #endif
    #if ALBEDO_MAP_DIM == 3
        float[3] P = float[3](_input[3], _input[4], _input[5]);
        forward(parameters.albedo, P, albedo);
    #endif
        _output[3] *= albedo[0];
        _output[4] *= albedo[1];
        _output[5] *= albedo[2];
    }}
        """
    )

    def __init__(self, base: SurfaceScatteringSampler, albedo: MapBase):
        assert albedo.input_dim == 2 or albedo.input_dim == 3, 'Can not use a map to albedo that is not 2D or 3D'
        assert albedo.output_dim == 3
        super().__init__(ALBEDO_MAP_DIM = albedo.input_dim)
        self.base_scattering_sampler = base
        self.albedo = albedo

    def get_scattering(self):
        return AlbedoSurfaceScattering(self.base_scattering_sampler.get_scattering(), self.albedo)

    def get_pdf(self):
        return self.base_scattering_sampler.get_pdf()


class MixtureSurfaceScatter(SurfaceScattering):
    __extension_info__ = dict(
        parameters=dict(
            scattering_a=MapBase,
            scattering_b=MapBase,
            alpha=MapBase
        ),
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=3),
        nodiff=True,
        code=f"""
    FORWARD {{
        float alpha[1];
    #if ALPHA_MAP_DIM == 2
        float[2] C = float[2](_input[15],_input[16]);
        forward(parameters.alpha, C, alpha);
    #endif
    #if ALPHA_MAP_DIM == 3
        float[3] P = float[3](_input[6], _input[7], _input[8]);
        forward(parameters.alpha, P, alpha);
    #endif
        float Wa[3];
        forward(parameters.scattering_a, _input, Wa);
        float Wb[3];
        forward(parameters.scattering_b, _input, Wb);
        _output = float[3](mix(Wa[0], Wb[0], alpha[0]), mix(Wa[1], Wb[1], alpha[0]), mix(Wa[2], Wb[2], alpha[0]));
    }}
        """
    )
    def __init__(self, scattering_a: SurfaceScattering, scattering_b: SurfaceScattering, alpha: MapBase):
        assert alpha.input_dim == 2 or alpha.input_dim == 3
        assert alpha.output_dim == 1
        super().__init__(ALPHA_MAP_DIM=alpha.input_dim)
        self.scattering_a = scattering_a
        self.scattering_b = scattering_b
        self.alpha = alpha


class MixtureSurfaceScatterSamplerPDF(SurfaceScatteringSamplerPDF):
    __extension_info__ = dict(
        parameters=dict(
            scattering_a_pdf=MapBase,
            scattering_b_pdf=MapBase,
            alpha=MapBase
        ),
        generics=dict(INPUT_DIM=23, OUTPUT_DIM=1),
        nodiff=True,
        code=f"""
    FORWARD {{
        float alpha[1];
    #if ALPHA_MAP_DIM == 2
        float[2] C = float[2](_input[15],_input[16]);
        forward(parameters.alpha, C, alpha);
    #endif
    #if ALPHA_MAP_DIM == 3
        float[3] P = float[3](_input[6], _input[7], _input[8]);
        forward(parameters.alpha, P, alpha);
    #endif
        float pdf_a[1];
        forward(parameters.scattering_a_pdf, _input, pdf_a);
        float pdf_b[1];
        forward(parameters.scattering_b_pdf, _input, pdf_b);
        _output[0] = mix(pdf_a[0], pdf_b[0], alpha[0]);
    }}
        """
    )
    def __init__(self, scattering_a_pdf: SurfaceScatteringSamplerPDF, scattering_b_pdf: SurfaceScatteringSamplerPDF, alpha: MapBase):
        assert alpha.input_dim == 2 or alpha.input_dim == 3
        assert alpha.output_dim == 1
        super().__init__(ALPHA_MAP_DIM=alpha.input_dim)
        self.scattering_a_pdf = scattering_a_pdf
        self.scattering_b_pdf = scattering_b_pdf
        self.alpha = alpha


class MixtureSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = dict(
        parameters=dict(
            scattering_sampler_a=MapBase,
            scattering_sampler_b=MapBase,
            alpha=MapBase
        ),
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        nodiff=True,
        code=f"""
        FORWARD {{
            float alpha[1];
        #if ALPHA_MAP_DIM == 2
            float[2] C = float[2](_input[12],_input[13]);
            forward(parameters.alpha, C, alpha);
        #endif
        #if ALPHA_MAP_DIM == 3
            float[3] P = float[3](_input[3], _input[4], _input[5]);
            forward(parameters.alpha, P, alpha);
        #endif
            if (random() < alpha[0]) // sample from a
                forward(parameters.scattering_sampler_a, _input, _output);
            else  // sample from b
                forward(parameters.scattering_sampler_b, _input, _output);
        }}
            """
    )
    def __init__(self, scattering_sampler_a: SurfaceScatteringSampler, scattering_sampler_b: SurfaceScatteringSampler, alpha: MapBase):
        assert alpha.input_dim == 2 or alpha.input_dim == 3
        assert alpha.output_dim == 1
        super().__init__(ALPHA_MAP_DIM=alpha.input_dim)
        self.scattering_sampler_a = scattering_sampler_a
        self.scattering_sampler_b = scattering_sampler_b
        self.alpha = alpha

    def get_scattering(self):
        return MixtureSurfaceScatter(
            self.scattering_sampler_a.get_scattering(),
            self.scattering_sampler_b.get_scattering(),
            self.alpha
        )

    def get_pdf(self):
        return MixtureSurfaceScatterSamplerPDF(
            self.scattering_sampler_a.get_pdf(),
            self.scattering_sampler_b.get_pdf(),
            self.alpha
        )


class NullSurfaceScatteringSampler(SurfaceScatteringSampler, Singleton):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        nodiff=True,
        code=f"""
    FORWARD {{
        _output = float[7](_input[0], _input[1], _input[2], 1.0, 1.0, 1.0, -1.0);
    }}
        """
    )
    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class NoSurfaceScatteringSampler(SurfaceScatteringSampler):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=7),
        nodiff=True,
        code=f"""
    FORWARD {{
        _output = float[7](0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }}
        """
    )

    def get_scattering(self):
        return DeltaSurfaceScattering.get_instance()

    def get_pdf(self):
        return DeltaSurfaceScatteringPDF.get_instance()


class SurfaceEmission(MapBase):
    __extension_info__ = None  # Abstract Node

    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: Optional[str] = None,
            external_code: Optional[str] = None,
            parameters: Optional[Dict] = None,
    ):
        if parameters is None:
            parameters = { }
        if bw_code is None:
            bw_code = ''
        if external_code is None:
            external_code = ''
        return dict(
            generics=dict(INPUT_DIM=20, OUTPUT_DIM=3),
            parameters=parameters,
            code = f"""
{external_code if external_code is not None else ''}

FORWARD {{
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    Surfel surfel = Surfel(
        vec3(_input[3], _input[4], _input[5]),
        vec3(_input[6], _input[7], _input[8]),
        vec3(_input[9], _input[10], _input[11]),
        vec2(_input[12], _input[13]),
        vec3(_input[14], _input[15], _input[16]),
        vec3(_input[17], _input[18], _input[19]));
    bool from_outside = dot(win, surfel.G) < 0;
    bool correct_hemisphere = (dot(win, surfel.N) < 0) == from_outside;
    vec3 N = correct_hemisphere ? surfel.N : surfel.G; 
    vec3 fN = from_outside ? N : -N;
    vec3 E;
    {fw_code}
    _output = float[3](E.x, E.y, E.z); 
}}

BACKWARD {{
    
}}
            """,

        )


class IsotropicSurfaceEmission(SurfaceEmission):
    __extension_info__ = SurfaceEmission.create_extension_info(
        f"""
        #if EMISSION_MAP_DIM == 2
        forward(parameters.emission_map, float[2](_input[12], _input[13]), _output);    
        #endif
        #if EMISSION_MAP_DIM == 3
        forward(parameters.emission_map, float[3](_input[3], _input[4], _input[5]), _output);    
        #endif    
        float weight = 0.5 / pi;
        E = vec3(_output[0], _output[1], _output[2])*weight;
        """,
        parameters=dict(
            emission_map=MapBase
        )
    )
    def __init__(self, emission_map: MapBase):
        assert emission_map.input_dim == 2 or emission_map.input_dim == 3
        assert emission_map.output_dim == 3
        super().__init__(EMISSION_MAP_DIM = emission_map.input_dim)
        self.emission_map = emission_map


class LambertSurfaceEmission(SurfaceEmission):
    __extension_info__ = SurfaceEmission.create_extension_info(
        f"""
        #if EMISSION_MAP_DIM == 2
        forward(parameters.emission_map, float[2](_input[12], _input[13]), _output);    
        #endif
        #if EMISSION_MAP_DIM == 3
        forward(parameters.emission_map, float[3](_input[3], _input[4], _input[5]), _output);    
        #endif
        float weight = dot(-win, fN) / pi;
        E = vec3(_output[0], _output[1], _output[2])*weight;
        """,
        parameters=dict(
            emission_map=MapBase
        )
    )
    def __init__(self, emission_map: MapBase):
        assert emission_map.input_dim == 2 or emission_map.input_dim == 3
        assert emission_map.output_dim == 3
        super().__init__(EMISSION_MAP_DIM = emission_map.input_dim)
        self.emission_map = emission_map


class NoSurfaceEmission(SurfaceEmission):
    __extension_info__ = dict(
        generics=dict(INPUT_DIM=20, OUTPUT_DIM=3),
        nodiff=True,
        code=f"""
    FORWARD {{
        _output = float[3](0.0, 0.0, 0.0);
    }}
        """
    )

    __instance__ = None
    @classmethod
    def get_instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = NoSurfaceEmission()
        return cls.__instance__


class SurfaceGathering(MapBase): # win, P, N, G, C, T, B -> R
    __extension_info__ = None

    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: Optional[str] = None,
            parameters: Optional[Dict] = None,
            external_code: Optional[str] = None, **kwargs
    ):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters  = {}
        if external_code is None:
            external_code = ''
        return dict(
            parameters=parameters,
            generics=dict(INPUT_DIM=20, OUTPUT_DIM=3),
            code=f"""
    {external_code}
    
    FORWARD {{
        vec3 win = vec3(_input[0], _input[1], _input[2]);
        Surfel surfel = Surfel(
            vec3(_input[3], _input[4], _input[5]),
            vec3(_input[6], _input[7], _input[8]),
            vec3(_input[9], _input[10], _input[11]),
            vec2(_input[12], _input[13]),
            vec3(_input[14], _input[15], _input[16]),
            vec3(_input[17], _input[18], _input[19]));
        bool from_outside = dot(win, surfel.G) < 0;
        bool correct_hemisphere = (dot(win, surfel.N) < 0) == from_outside;
        vec3 N = correct_hemisphere ? surfel.N : surfel.G; 
        vec3 fN = from_outside ? N : -N;
        vec3 A = vec3(0.0);
        {fw_code}
        _output = float[3](A.x, A.y, A.z);
    }}
    
    BACKWARD {{
        {bw_code}
    }}
            """,
            **kwargs
        )

    @staticmethod
    def signature():
        return (20, 3)


class NoSurfaceGathering(SurfaceGathering, Singleton):
    __extension_info__ = {
        **SurfaceGathering.create_extension_info(''),
        'code': f"""
    FORWARD {{
        _output = float[3](0.0, 0.0, 0.0);
    }}
    BACKWARD {{ }}
    """
    }


class EnvironmentGathering(SurfaceGathering):
    __extension_info__ = SurfaceGathering.create_extension_info(
        """
        // Check scattering if 
        // x, s -> w, W, pdf
        vec3 x;
        float[7] ss_values;
        forward(parameters.scattering_sampler, _input, ss_values);

        vec3 we = vec3(ss_values[0], ss_values[1], ss_values[2]);
        vec3 E, W;
        if (ss_values[6] == -1) // delta scattering
        {
            x = surfel.P + we * 0.0001;
            environment(object, x, we, E);
            W = vec3(ss_values[3], ss_values[4], ss_values[5]);
        }
        else 
        {
            x = fN * 0.0001 + surfel.P;
            float pdf;
            environment_sampler(object, x, we, E, pdf);
            if (E == vec3(0)) {
                _output = float[3](0.0, 0.0, 0.0);
                return;
            }
        
            float[3] W_values;
            float[23] s_input = float[23](_input[0], _input[1], _input[2], we.x, we.y, we.z, 
            _input[3], _input[4], _input[5], _input[6], _input[7], _input[8],
             _input[9], _input[10], _input[11], 
             _input[12], _input[13], 
             _input[14], _input[15], _input[16],
             _input[17], _input[18], _input[19]); 
            forward(parameters.scattering, s_input, W_values);
            W = vec3(W_values[0], W_values[1], W_values[2]);
            if (W == vec3(0))
            {
                _output = float[3](0.0, 0.0, 0.0);
                return;
            }
        }
        float Tr = ray_visibility(object, x, we); 
        A += Tr * W * E; 
        """,
        parameters=dict(
            environment=MapBase,
            environment_sampler=MapBase,
            visibility=MapBase,
            scattering=MapBase,
            scattering_sampler=MapBase,
        ),
        external_code="""
    #include "sc_environment_sampler.h"
    #include "sc_environment.h"
    #include "sc_visibility.h"
        """
    )

    def __init__(self, environment_sampler: EnvironmentSampler, visibility: MapBase, surface_scattering_sampler: Optional[SurfaceScatteringSampler] = None):
        super().__init__()
        self.environment = environment_sampler.get_environment()
        self.environment_sampler = environment_sampler
        self.visibility = visibility
        self.scattering = constant(23, 0.0, 0.0, 0.0) if surface_scattering_sampler is None else surface_scattering_sampler.get_scattering()
        self.scattering_sampler = constant(20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0) if surface_scattering_sampler is None else surface_scattering_sampler


class SurfaceMaterial:
    def __init__(self, sampler: Optional[SurfaceScatteringSampler] = None, emission: Optional[SurfaceEmission] = None):
        self.sampler = sampler
        self.sampler_pdf = None if sampler is None else sampler.get_pdf()
        self.scattering = None if sampler is None else sampler.get_scattering()
        self.emission = emission

    def emission_gathering(self):
        return self.emission

    def environment_gathering(self, environment_sampler: EnvironmentSampler, visibility: MapBase):
        return None if self.scattering is None else (
            EnvironmentGathering(
                environment_sampler,
                visibility,
                self.sampler
            ))

    def environment_and_emission_gathering(self, environment_sampler: EnvironmentSampler, visibility: MapBase):
        env = self.environment_gathering(environment_sampler, visibility)
        em = self.emission_gathering()
        if em is None:
            return env
        if env is None:
            return em
        return env + em

    def photon_gathering(self, incomming_radiance: MapBase):
        pass


class MediumPathIntegrator(MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: Optional[str] = None,
            parameters: Optional[Dict] = None,
            external_code: Optional[str] = None
    ):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        parameters.update(dict(gathering=MapBase))
        if external_code is None:
            external_code = ''

        return dict(
            parameters = parameters,
            # dynamics=[(6, 3)],
            generics = dict(INPUT_DIM=7, OUTPUT_DIM=13),
            code=f"""
    
    vec3 gathering(map_object, vec3 xs, vec3 ws)
    {{
        float _output[3];
        forward(parameters.gathering, float[6](xs.x, xs.y, xs.z, ws.x, ws.y, ws.z), _output);
        return vec3(_output[0], _output[1], _output[2]); 
    }}
    
    /* vec3 gathering(map_object, vec3 xs, vec3 ws)
    {{
        if (parameters.gathering == 0)
        return vec3(0.0);

        float _output[3];
        dynamic_forward(object, parameters.gathering, float[6](xs.x, xs.y, xs.z, ws.x, ws.y, ws.z), _output);
        return vec3(_output[0], _output[1], _output[2]); 
    }}*/
    
    {external_code}
    
    FORWARD {{
        vec3 x = vec3(_input[0], _input[1], _input[2]);
        vec3 w = vec3(_input[3], _input[4], _input[5]);
        float d = _input[6];
        vec3 xo, wo;
        float Tr = 1.0;
        vec3 W = vec3(0.0);
        vec3 A = vec3(0.0);
        {fw_code}
        /*
        #if MEDIUM_FILTER & 1 == 0
        W = vec3(0.0);
        #endif 
        #if MEDIUM_FILTER & 2 == 0
        A = vec3(0.0);
        #endif
        */
        _output = float[13](
            Tr,
            xo.x, xo.y, xo.z,
            wo.x, wo.y, wo.z,
            W.x, W.y, W.z,
            A.x, A.y, A.z
        );
    }}
    
    BACKWARD {{
        {bw_code}
    }}
            """
        )

    @staticmethod
    def signature():
        return (7, 13)

    def __init__(self, gathering: Optional[MapBase] = None, medium_filter: int = 3):
        if gathering is None:
           medium_filter &= 1  # only scatter
        super().__init__(MEDIUM_FILTER = medium_filter)
        # self.gathering_module = gathering
        # self.gathering = 0 if gathering is None else gathering.__bindable__.device_ptr # constant(6, 0., 0., 0.) if gathering is None else gathering
        self.gathering = constant(6, 0.0, 0.0, 0.0) if gathering is None else gathering


class DTMediumPathIntegrator(MediumPathIntegrator):
    __extension_info__ = MediumPathIntegrator.create_extension_info(
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            phase_sampler=MapBase,
            majorant=ParameterDescriptor
        ),
        external_code=f"""
    #include "vr_sigma.h"
    #include "vr_scattering_albedo.h"
    #include "sc_phase_sampler.h"
        """,
        fw_code="""
    float majorant = max(0.00001, param_float(parameters.majorant));
    while(true) 
    {
        float t = -log(1 - random()) / majorant;
        if (t > d - 0.00001) // exit
            break;
        x += t * w;
        d -= t;
        float s = sigma(object, x);
        if (random() < s / majorant) // interaction
        {
    #if MEDIUM_FILTER & 2 == 2
            A = gathering(object, x, w);
    #endif
            Tr = 0.0;
    #if MEDIUM_FILTER & 1 == 1
            xo = x;
            float weight, rho;
            wo = sample_phase(object, w, weight, rho);
            W = scattering_albedo(object, x);
    #endif
            break;
        }
    }
    """)

    def __init__(self, sigma: MapBase, majorant: torch.Tensor, scattering_albedo: Optional[MapBase] = None, phase_sampler: Optional[MapBase] = None, gathering: Optional[MapBase] = None, medium_filter: int = 3):
        if scattering_albedo is None or phase_sampler is None:
            medium_filter &= 2
            scattering_albedo = constant(3, 0.0, 0.0, 0.0)
            phase_sampler = constant(3, 0.0, 0.0, 0.0, 0.0, 0.0)
        super().__init__(gathering, medium_filter=medium_filter)
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.phase_sampler = phase_sampler
        self.majorant = parameter(majorant)


class HomogeneousMediumPathIntegrator(MediumPathIntegrator):
    __extension_info__ = MediumPathIntegrator.create_extension_info(
        parameters=dict(
            sigma=ParameterDescriptor,
            scattering_albedo=ParameterDescriptor,
            phase_sampler=MapBase
        ),
        external_code=f"""
        #include "sc_phase_sampler.h"
            """,
        fw_code="""
        float sigma = param_float(parameters.sigma);
        Tr = exp(-d * sigma);
        float t = -log(1 - random()*(1 - Tr)) / max(0.0000001, sigma);
        //t = min(t, d);
        xo = x + w * t;
        #if MEDIUM_FILTER & 2
        A = gathering(object, xo, w);
        #endif
        #if MEDIUM_FILTER & 1
        vec3 albedo = param_vec3(parameters.scattering_albedo);
        float weight, pdf;
        wo = sample_phase(object, w, weight, pdf); 
        W = albedo * weight;
        #endif
        """)
    def __init__(self, sigma: torch.Tensor, scattering_albedo: Optional[torch.Tensor] = None,
                 phase_sampler: Optional[MapBase] = None, gathering: Optional[MapBase] = None, medium_filter: int = 3):
        if scattering_albedo is None or phase_sampler is None:
            medium_filter &= 2
            scattering_albedo = torch.zeros(3, device=device())
            phase_sampler = constant(3, 0.0, 0.0, 0.0, 0.0, 0.0)
        super().__init__(gathering, medium_filter)
        self.sigma = parameter(sigma)
        self.scattering_albedo = parameter(scattering_albedo)
        self.phase_sampler = phase_sampler


# class VolumeEmissionGathering(MapBase):
#     __extension_info__ = dict(
#         generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
#         dynamics=[(3, 3), (6, 3)],
#         parameters=dict(
#             scattering_albedo=torch.int64,
#             emission=torch.int64
#         ),
#         code=f"""
#     FORWARD {{
#         _output = float[3](0.0, 0.0, 0.0);
#         vec3 E = vec3(0.0);
#         if (parameters.emission != 0)
#         {{
#             float [3] em;
#             dynamic_forward(object, parameters.emission, _input, em);
#             E = vec3(em[0], em[1], em[2]);
#         }}
#         else
#         return;
#
#         vec3 scattering_albedo = vec3(0.0);
#         if (parameters.scattering_albedo != 0)
#         {{
#             float[3] sa;
#             dynamic_forward(object, parameters.scattering_albedo, float[3](_input[0], _input[1], _input[2]), sa);
#             scattering_albedo = vec3(sa[0], sa[1], sa[2]);
#         }}
#
#         vec3 o = (vec3(1.0) - scattering_albedo)*E;
#         _output = float[3](o.x, o.y, o.z);
#     }}
#
#     BACKWARD {{  }}
#         """
#     )
#
#     def __init__(self, scattering_albedo: Optional[MapBase], emission: Optional[MapBase]):
#         super().__init__()
#         self.scattering_albedo_module = scattering_albedo
#         self.scattering_albedo = 0 if scattering_albedo is None else scattering_albedo.__bindable__.device_ptr
#         self.emission_module = emission
#         self.emission = 0 if emission is None else emission.__bindable__.device_ptr


class VolumeGathering(MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(
            fw_code: str,
            bw_code: Optional[str] = None,
            parameters: Optional[Dict] = None,
            external_code: Optional[str] = None, **kwargs
    ):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        if external_code is None:
            external_code = ''
        return dict(
            parameters=parameters,
            generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
            code=f"""
       {external_code}

       FORWARD {{
           vec3 x = vec3(_input[0], _input[1], _input[2]);
           vec3 w = vec3(_input[3], _input[4], _input[5]);
           vec3 A = vec3(0.0);
           {fw_code}
           _output = float[3](A.x, A.y, A.z);
       }}

       BACKWARD {{
           {bw_code}
       }}
               """,
            **kwargs
        )

    @staticmethod
    def signature():
        return (6, 3)


class VolumeEmissionGathering(VolumeGathering):
    __extension_info__ = VolumeGathering.create_extension_info(
        parameters=dict(
            scattering_albedo=MapBase,
            emission=MapBase
        ),
        fw_code=f"""
    forward(parameters.emission, _input, _output);
    float[3] sa;
    forward(parameters.scattering_albedo, float[3](_input[0], _input[1], _input[2]), sa);
    _output[0] *= 1 - sa[0];
    _output[1] *= 1 - sa[1];
    _output[2] *= 1 - sa[2];
    return;
        """
    )

    def __init__(self, scattering_albedo: Optional[MapBase], emission: Optional[MapBase]):
        super().__init__()
        self.scattering_albedo = constant(3, 0.0, 0.0, 0.0) if scattering_albedo is None else scattering_albedo
        self.emission = constant(6, 0.0, 0.0, 0.0) if emission is None else emission


class VolumeEnvironmentGathering(VolumeGathering):
    __extension_info__ = VolumeGathering.create_extension_info(
        parameters=dict(
            environment_sampler=MapBase,
            visibility=MapBase,
            scattering_albedo=MapBase,
            phase=MapBase,
        ),
        external_code="""
        #include "vr_scattering_albedo.h"
        #include "sc_visibility.h"
        #include "sc_environment_sampler.h"
        """,
        fw_code="""
        
        vec3 we, E;
        float pdf;
        environment_sampler(object, x, we, E, pdf);
        //E = vec3(1.0);
        //pdf = 1.0;
        //we = randomDirection(w);
        
        if (E == vec3(0.0))
        {
            _output = float[3](0.0, 0.0, 0.0);
            return;
        }
    
        vec3 W = scattering_albedo(object, x);

        float rho[1];
        forward(parameters.phase, float[6](w.x, w.y, w.z, we.x, we.y, we.z), rho);
        W *= rho[0];

        if (W == vec3(0.0))
        {
            _output = float[3](0.0, 0.0, 0.0);
            return;
        }
        
        float Tr = ray_visibility(object, x, we);
        A += Tr * W * E; 
             """
    )

    def __init__(self, environment_sampler: EnvironmentSampler, visibility: MapBase, scattering_albedo: Optional[MapBase], phase: Optional[MapBase]):
        super().__init__()
        self.environment_sampler = environment_sampler
        self.visibility = visibility
        self.scattering_albedo = constant(3, 0.0, 0.0, 0.0) if scattering_albedo is None else scattering_albedo
        self.phase = constant(6, 0.0) if phase is None else phase



# class VolumeNEEGathering(MapBase):
#     __extension_info__ = dict(
#         generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
#         dynamics=[(3, 3), (6, 3)],
#         parameters=dict(
#             scattering_albedo=torch.int64,
#             emission=torch.int64
#         ),
#         code=f"""
#         FORWARD {{
#             if (parameters.emission == 0)
#             {{
#                 _output = float[3](0.0, 0.0, 0.0);
#                 return;
#             }}
#             vec3 scattering_albedo = vec3(0.0);
#             if (parameters.scattering_albedo != 0)
#             {{
#                 float[3] sa;
#                 dynamic_forward(object, parameters.scattering_albedo, float[3](_input[0], _input[1], _input[2]), sa);
#                 scattering_albedo = vec3(sa[0], sa[1], sa[2]);
#             }}
#             vec3 o = vec3(1.0) - scattering_albedo;
#             float[3] em;
#             dynamic_forward(object, parameters.emission, _input, em);
#             o *= vec3(em[0], em[1], em[2]);
#             _output = float[3](o.x, o.y, o.z);
#         }}
#             """
#     )
#
#     def __init__(self, scattering_albedo: Optional[MapBase], phase_sampler: Optional[MapBase], emission: Optional[MapBase]):
#         super().__init__()
#         self.scattering_albedo_module = scattering_albedo
#         self.scattering_albedo = 0 if scattering_albedo is None else scattering_albedo.__bindable__.device_ptr
#         self.emission_module = emission
#         self.emission = 0 if emission is None else emission.__bindable__.device_ptr


class PhaseSamplerPDF(MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(fw_code: str,
                              bw_code: Optional[str] = None,
                              parameters: Optional[Dict] = None):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        return dict(
            parameters=parameters,
            generics=dict(INPUT_DIM = 6, OUTPUT_DIM=1),
            code=f"""
    FORWARD {{
        vec3 win = vec3(_input[0], _input[1], _input[2]);
        vec3 wout = vec3(_input[3], _input[4], _input[5]);
        float pdf;
        {fw_code}
        _output[0] = pdf;
    }}
    BACKWARD {{
        {bw_code}
    }}
            """
        )


class PhaseSampler(MapBase):
    __extension_info__ = None
    @staticmethod
    def create_extension_info(fw_code: str, bw_code: Optional[str] = None, parameters: Optional[Dict]= None):
        if bw_code is None:
            bw_code = ''
        if parameters is None:
            parameters = {}
        return dict(
            generics=dict(INPUT_DIM=3, OUTPUT_DIM=5),
            parameters = parameters,
            code=f"""
    FORWARD {{
        vec3 win = vec3(_input[0], _input[1], _input[2]);
        vec3 wout; float weight; float pdf;
        {fw_code}
        _output = float[5](wout.x, wout.y, wout.z, weight, pdf);
    }}
    BACKWARD {{
        {bw_code}
    }}
            """
        )

    def get_pdf(self) -> PhaseSamplerPDF:
        raise NotImplementedError()

    def get_phase(self) -> MapBase:
        raise NotImplementedError()


class IsotropicPhaseSamplerPdf(PhaseSamplerPDF, Singleton):
    __extension_info__ = PhaseSamplerPDF.create_extension_info(f"""
    pdf = 0.25/pi;
    """)


class IsotropicPhaseSampler(PhaseSampler, Singleton):
    __extension_info__ = PhaseSampler.create_extension_info("""
    wout = randomDirection(win);
    pdf = 0.25 / pi;
    weight = 1.0;
    """)

    def get_pdf(self) -> PhaseSamplerPDF:
        return IsotropicPhaseSamplerPdf.get_instance()

    def get_phase(self) -> MapBase:
        return IsotropicPhaseSamplerPdf.get_instance()


class HGPhaseSamplerPDF(PhaseSamplerPDF):
    __extension_info__ = PhaseSamplerPDF.create_extension_info(
        parameters=dict(
            g=ParameterDescriptor
        ),
        fw_code="""
    float g = param_float(parameters.g);
    pdf = hg_phase_eval(win, wout, g);
    """)

    def __init__(self, g: torch.Tensor):
        super().__init__()
        self.g = parameter(g)


class HGPhaseSampler(PhaseSampler):
    __extension_info__ = PhaseSampler.create_extension_info(
        parameters=dict(
            g=ParameterDescriptor
        ),
        fw_code="""
    float g = param_float(parameters.g);
    wout = hg_phase_sample(win, g, pdf);
    weight = 1.0; // Importance sampled function
        """
    )
    def __init__(self, g: torch.Tensor):
        super().__init__()
        self.g = parameter(g)
        self.pdf = HGPhaseSamplerPDF(self.g)

    def get_pdf(self) -> PhaseSamplerPDF:
        return self.pdf

    def get_phase(self) -> MapBase:
        return self.get_pdf()  # perfect importance sampled



class MediumMaterial:
    def __init__(self, path_integrator_factory: Callable[[Optional[MapBase], int], MapBase], scattering_albedo: Optional[MapBase] = None, phase_sampler: Optional['PhaseSampler'] = None, emission: Optional[MapBase] = None, enclosed: bool = False):
        self.path_integrator_factory = path_integrator_factory
        self.scattering_albedo = scattering_albedo
        self.emission = emission
        self.phase_sampler = phase_sampler
        self.phase_sampler_pdf = None if phase_sampler is None else phase_sampler.get_pdf()
        self.enclosed = enclosed  # means that it will be assumed that the boundary it's not null

    def transmittance(self):
        return self.path_integrator_factory(None, 0)

    def only_emission(self):
        return self.path_integrator_factory(None if self.emission is None else VolumeEmissionGathering(self.scattering_albedo, self.emission), 1)

    def only_scattering(self):
        return self.path_integrator_factory(None, 2)

    def default_gathering(self):
        return self.path_integrator_factory(None if self.emission is None else VolumeEmissionGathering(self.scattering_albedo, self.emission), 3)

    def environment_gathering(self, environment_sampler: EnvironmentSampler, visibility: MapBase):
        nee_gathering = None if self.scattering_albedo is None or self.enclosed else VolumeEnvironmentGathering(
                                                environment_sampler,
                                                visibility,
                                                self.scattering_albedo,
                                                self.phase_sampler.get_phase()
                                            )
        if nee_gathering is None:
            gathering = self.emission
        elif self.emission is None:
            gathering = nee_gathering
        else:
            gathering = nee_gathering + self.emission
        return self.path_integrator_factory(gathering, 3)
        # return self.path_integrator_factory(None, 0)

    def photon_gathering(self, incomming_radiance: MapBase):
        pass


class HomogeneousMediumMaterial(MediumMaterial):
    def __init__(self, sigma: Union[torch.Tensor, float], scattering_albedo: Optional[torch.Tensor] = None, phase_sampler: Optional['PhaseSampler'] = None, emission: Optional[MapBase] = None, enclosed: bool = False):
        super().__init__(
            lambda gathering, filter: HomogeneousMediumPathIntegrator(
                sigma=sigma,
                scattering_albedo=scattering_albedo,
                phase_sampler=phase_sampler,
                gathering=gathering,
                medium_filter=filter
            ),
            scattering_albedo=None if scattering_albedo is None else constant(3, scattering_albedo),
            phase_sampler=phase_sampler,
            emission=emission,
            enclosed=enclosed
        )


class HeterogeneousMediumMaterial(MediumMaterial):
    def __init__(self, sigma: MapBase, scattering_albedo: Optional[MapBase] = None, emission: Optional[MapBase]=None, phase_sampler: Optional['PhaseSampler'] = None, technique: Literal['dt'] = 'dt', **kwargs):
        if technique == 'dt':
            assert 'majorant' in kwargs, "Delta tracking technique requires a majorant: torch.Tensor parameter"
            def factory(gathering, filter):
                return DTMediumPathIntegrator(
                    sigma=sigma,
                    scattering_albedo=scattering_albedo,
                    phase_sampler=phase_sampler,
                    majorant=kwargs.get('majorant'),
                    gathering=gathering,
                    medium_filter=filter
                )
        else:
            raise NotImplementedError()
        super().__init__(
            path_integrator_factory=factory,
            scattering_albedo=scattering_albedo,
            phase_sampler=phase_sampler,
            emission=emission
        )



PTPatchInfo = map_struct(
    'PTPatchInfo',
    surface_scattering_sampler=torch.int64,
    surface_gathering=torch.int64,
    inside_medium=torch.int64,
    outside_medium=torch.int64
)


VSPatchInfo = map_struct(
    'VSPatchInfo',
    inside_medium=torch.int64,
    outside_medium=torch.int64,
    surface_scatters = int,
    pad0 = int,
    pad1=int,
    pad2=int,
)


class SceneVisibility(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            boundaries=MapBase,
            patch_info=[-1, VSPatchInfo]
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=1),
        dynamics=[
            MediumPathIntegrator.signature()
        ],
        path=__INCLUDE_PATH__+'/maps/scene_visibility.h'
    )
    def __init__(self,
        boundaries: Boundary,
        surface_scatters: Optional[List[bool]] = None,
        medium_integrator: Optional[List[MediumPathIntegrator]] = None,
        inside_medium_indices: Optional[List[int]] = None,
        outside_medium_indices: Optional[List[int]] = None,
    ):
        super().__init__(len(boundaries))
        self.boundaries = boundaries
        for i in range(len(boundaries)):
            self.patch_info[i].surface_scatters = 0
            self.patch_info[i].inside_medium = 0
            self.patch_info[i].outside_medium = 0
        assert surface_scatters is None or len(surface_scatters) == len(boundaries)
        if surface_scatters is not None:
            for i, ss in enumerate(surface_scatters):
                self.patch_info[i].surface_scatters = 0 if not ss else 1
        self.media_modules = medium_integrator
        assert inside_medium_indices is None or len(inside_medium_indices) == len(boundaries)
        if inside_medium_indices is not None:
            for i, imi in enumerate(inside_medium_indices):
                assert imi == -1 or (medium_integrator is not None and imi < len(medium_integrator))
                self.patch_info[i].inside_medium = 0 if imi == -1 else medium_integrator[imi].__bindable__.device_ptr
        assert outside_medium_indices is None or len(outside_medium_indices) == len(boundaries)
        if outside_medium_indices is not None:
            for i, omi in enumerate(outside_medium_indices):
                assert omi == -1 or (medium_integrator is not None and omi < len(medium_integrator))
                self.patch_info[i].outside_medium = 0 if omi == -1 else medium_integrator[omi].__bindable__.device_ptr


class PathtracedScene(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            surfaces=MapBase,
            boundaries=MapBase,
            environment=MapBase,
            patch_info=[-1, PTPatchInfo]
        ),
        generics=dict(INPUT_DIM=6, OUTPUT_DIM=3),
        dynamics=[
            SurfaceScatteringSampler.signature(),
            SurfaceGathering.signature(),
            MediumPathIntegrator.signature()
        ],
        path=__INCLUDE_PATH__+'/maps/pathtraced_scene.h'
    )
    def __init__(self,
        surfaces: Geometry,
        environment: MapBase,
        surface_gathering: Optional[List[SurfaceGathering]] = None,
        surface_scattering_sampler: Optional[List[SurfaceScatteringSampler]] = None,
        medium_integrator: Optional[List[MediumPathIntegrator]] = None,
        inside_medium_indices: Optional[List[int]] = None,
        outside_medium_indices: Optional[List[int]] = None,
    ):
        super().__init__(len(surfaces))
        self.surfaces = surfaces
        self.boundaries = surfaces.get_boundary()
        self.environment = environment
        for i in range(len(surfaces)):
            self.patch_info[i].surface_scattering_sampler = 0
            self.patch_info[i].surface_gathering = 0
            self.patch_info[i].inside_medium = 0
            self.patch_info[i].outside_medium = 0
        assert surface_scattering_sampler is None or len(surface_scattering_sampler) == len(surfaces)
        self.surface_scatterers = surface_scattering_sampler
        if surface_scattering_sampler is not None:
            for i, sss in enumerate(surface_scattering_sampler):
                self.patch_info[i].surface_scattering_sampler = 0 if sss is None else sss.__bindable__.device_ptr
        self.surface_gatherers = surface_gathering
        if surface_gathering is not None:
            for i, sg in enumerate(surface_gathering):
                self.patch_info[i].surface_gathering = 0 if sg is None else sg.__bindable__.device_ptr
        self.media_modules = medium_integrator
        assert inside_medium_indices is None or len(inside_medium_indices) == len(surfaces)
        if inside_medium_indices is not None:
            for i, imi in enumerate(inside_medium_indices):
                assert imi == -1 or (medium_integrator is not None and imi < len(medium_integrator))
                self.patch_info[i].inside_medium = 0 if imi == -1 else medium_integrator[imi].__bindable__.device_ptr
        if outside_medium_indices is not None:
            for i, omi in enumerate(outside_medium_indices):
                assert omi == -1 or (medium_integrator is not None and omi < len(medium_integrator))
                self.patch_info[i].outside_medium = 0 if omi == -1 else medium_integrator[omi].__bindable__.device_ptr


class Scene:
    def __init__(self,
                 surfaces: Geometry,
                 environment_sampler: EnvironmentSampler,
                 materials: Optional[List[SurfaceMaterial]] = None,
                 inside_media: Optional[List[MediumMaterial]] = None,
                 outside_media: Optional[List[MediumMaterial]] = None
                 ):
        self.surfaces = surfaces
        self.environment_sampler = environment_sampler
        self.materials = materials
        self.media = []
        if inside_media is not None:
            for m in inside_media:
                if m is not None and m not in self.media:
                    self.media.append(m)
        if outside_media is not None:
            for m in outside_media:
                if m is not None and m not in self.media:
                    self.media.append(m)
        self.inside_media = None if inside_media is None else [-1 if m is None else self.media.index(m) for m in inside_media]
        self.outside_media = None if outside_media is None else [-1 if m is None else self.media.index(m) for m in outside_media]
        self.scene_visibility = SceneVisibility(
            boundaries=self.surfaces.get_boundary(),
            surface_scatters=None if self.materials is None else [
                False if m is None or isinstance(m, NoSurfaceScatteringSampler) else True for m in self.materials
            ],
            medium_integrator=None if self.media is None else [
                None if m is None else m.transmittance() for m in self.media
            ],
            inside_medium_indices=self.inside_media,
            outside_medium_indices=self.outside_media
        )

    @staticmethod
    def from_graph(
            geometries: Dict[str, Geometry],
            environment_sampler: EnvironmentSampler,
            materials: Optional[Dict[str, SurfaceMaterial]] = None,
            inside_media: Optional[Dict[str, MediumMaterial]] = None,
            outside_media: Optional[Dict[str, MediumMaterial]] = None
    ):
        # if len(geometries) == 1:
        #     surfaces = next(iter(geometries.values()))
        # else:
        surfaces = GroupGeometry(*(v for k, v in geometries.items()))
        ranges = {}
        offset = 0
        for k, v in geometries.items():
            ranges[k] = (offset, offset + len(v))
            offset += len(v)
        if materials is None:
            surface_materials = None
        else:
            surface_materials = [None] * offset
            for k, m in materials.items():
                if k in geometries:
                    for i in range(*ranges[k]):
                        surface_materials[i] = m
        if inside_media is None:
            scene_inside_media = None
        else:
            scene_inside_media = [None] * offset
            for k, m in inside_media.items():
                if k == '__all__':
                    for i in range(offset):
                        scene_inside_media[i] = m
                elif k in geometries:
                    for i in range(*ranges[k]):
                        scene_inside_media[i] = m
        if outside_media is None:
            scene_outside_media = None
        else:
            scene_outside_media = [None] * offset
            for k, m in outside_media.items():
                if k == '__all__':
                    for i in range(offset):
                        scene_outside_media[i] = m
                elif k in geometries:
                    for i in range(*ranges[k]):
                        scene_outside_media[i] = m
        return Scene (surfaces, environment_sampler, surface_materials, scene_inside_media, scene_outside_media)

    def visibility(self) -> MapBase:
        """
        Creates a visibility map using the scene surfaces and transmittance of volumes
        """
        return self.scene_visibility


    def pathtrace(self) -> MapBase:
        return PathtracedScene(
            surfaces=self.surfaces,
            environment=self.environment_sampler.get_environment(),
            surface_gathering=None if self.materials is None else [
                None if m is None else m.emission_gathering() for m in self.materials
            ],
            surface_scattering_sampler=None if self.materials is None else [
                None if m is None else m.sampler for m in self.materials
            ],
            medium_integrator=None if self.media is None else [
                None if m is None else m.default_gathering() for m in self.media
            ],
            inside_medium_indices=self.inside_media,
            outside_medium_indices=self.outside_media
        )

    def pathtrace_environment_nee(self) -> MapBase:
        return self.transmittance() + PathtracedScene(
            surfaces=self.surfaces,
            # environment=ray_direction(),
            environment=constant(6, 0.0, 0.0, 0.0),
            surface_gathering=None if self.materials is None else [
                # None if m is None else m.emission_gathering() for m in self.materials
                # None for m in self.materials
                None if m is None else m.environment_gathering(self.environment_sampler, self.visibility()) for m in self.materials
            ],
            surface_scattering_sampler=None if self.materials is None else [
                None if m is None else m.sampler for m in self.materials
            ],
            medium_integrator=None if self.media is None else [
                None if m is None else m.environment_gathering(self.environment_sampler, self.visibility()) for m in self.media
                # None if m is None else m.default_gathering() for m in self.media
        ],
            inside_medium_indices=self.inside_media,
            outside_medium_indices=self.outside_media
        )

    def transmittance(self) -> MapBase:
        return self.visibility().after(ray_to_segment(constant(6, 10000.0))).promote(3) * self.environment_sampler.get_environment()


class Medium(MapBase):
    __extension_info__ = None  # Abstract Node
    @staticmethod
    def create_extension_info(
        fw_code: str,
        bw_code: Optional[str] = None,
        external_code: Optional[str] = None,
        parameters: Optional[Dict] = None
    ):
        return dict(
            generics=dict(INPUT_DIM=7, OUTPUT_DIM=13),
            parameters=parameters if parameters is not None else {},
            code=f"""
{external_code if external_code is not None else ''}

FORWARD {{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float Tr = 1.0;
    vec3 xout, wout, W = vec3(0.0), A = vec3(0.0); 
       
    {fw_code}
    
    _output = float[13] (Tr, xout.x, xout.y, xout.z, wout.x, wout.y, wout.z, W.x, W.y, W.z, A.x, A.y, A.z);
}}

BACKWARD {{
    {bw_code if bw_code is not None else ''}
}}
            """
        )

    def __init__(self):
        super().__init__()  # x, w, d -> T, xe, we, W, A


class NoMedium(Medium):
    __extension_info__ = Medium.create_extension_info(f"""
        Tr = 1.0; // full transmittance
        W = vec3(0.0); // no scattering
        A = vec3(0.0); // no accumulation
        xout = w * d + x;
        wout = w;
    """)

    __instance__ = None
    @classmethod
    def get_instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = NoMedium()
        return cls.__instance__


class HomogeneousEmissionMedium(Medium):
    __extension_info__ = Medium.create_extension_info(
        parameters=dict(
            sigma=ParameterDescriptor,
            emission=ParameterDescriptor,
        ),
        fw_code=f"""
    float sigma = param_float(parameters.sigma);
    vec3 emission = param_vec3(parameters.emission);
    Tr = exp(-d * sigma);
    W = vec3(0.0);
    A = (1 - Tr) * emission; 
    """)

    def __init__(self, sigma: torch.Tensor, emission: torch.Tensor):
        super().__init__()
        self.sigma = parameter(sigma)
        self.emission = parameter(emission)





class HomogeneousMedium(Medium):
    __extension_info__ = Medium.create_extension_info(
        parameters=dict(
            sigma=ParameterDescriptor,
            scattering_albedo=ParameterDescriptor,
            phase_sampler=MapBase
        ),
        external_code="#include \"sc_phase_sampler.h\"",
        fw_code=f"""
        float sigma = param_float(parameters.sigma);
        vec3 albedo = param_vec3(parameters.scattering_albedo);
        Tr = exp(-d * sigma);
        float t = -log(1 - random()*(1 - Tr)) / max(0.0000001, sigma);
        xout = x + w * t;
        float weight, pdf;
        wout = sample_phase(object, w, weight, pdf); 
        W = albedo * weight;
        """)

    def __init__(self, sigma: torch.Tensor, scattering_albedo: torch.Tensor, phase_sampler: PhaseSampler):
        super().__init__()
        self.sigma = parameter(sigma)
        self.scattering_albedo = parameter(scattering_albedo)
        self.phase_sampler = phase_sampler


class DTHeterogeneousMedium(Medium):
    __extension_info__ = Medium.create_extension_info(
        parameters=dict(
            sigma=MapBase,
            scattering_albedo=MapBase,
            phase_sampler=MapBase,
            majorant=ParameterDescriptor,
        ),
        external_code=f"""
        #include "sc_phase_sampler.h"
        #include "vr_sigma.h"
        #include "vr_scattering_albedo.h"
        """,
        fw_code=f"""
    float m = max(0.0001, param_float(parameters.majorant));
    A = vec3(0.0);
    W = vec3(0.0);
    Tr = 1.0;
    while (true)
    {{
        float t = -log(1 - random()) / m;
        if (t > d - 0.00001)
            break;
        x += w * t;
        float s = sigma(object, x); 
        if (random() < s/m) // real interaction
        {{
            float weight, pdf;
            wout = sample_phase(object, w, weight, pdf); 
            W = scattering_albedo(object, x) * weight;
            xout = x;
            Tr = 0.0;
            break;
        }}
        d -= t;
    }}
        """
    )

    def __init__(self, sigma: MapBase, scattering_albedo: MapBase, phase_sampler: MapBase, majorant: torch.Tensor):
        super().__init__()
        self.sigma = sigma
        self.scattering_albedo = scattering_albedo
        self.phase_sampler = phase_sampler
        self.majorant = parameter(majorant)


class EnvironmentEmission(MapBase):
    __extension_info__ = None
    def __init__(self):
        super().__init__(INPUT_DIM=3, OUTPUT_DIM=3)  # w -> R


class Pathtracer(MapBase):
    __extension_info__ = dict(
        parameters=dict(
            surfaces=MapBase,  # Scene to raycast
            environment=MapBase,  # Radiance map directional emitted from infinity
            surface_emission=torch.Tensor,
            surface_scattering_sampler=torch.Tensor,
            inside_media=torch.Tensor,
            outside_media=torch.Tensor,
        ),
        dynamics=[
            (20, 3),  #surface emitters
            (20, 7),  #surface scattering samplers
            (7, 13),  #volume scattering
        ],
        generics=dict(
            INPUT_DIM=6,  # x, w
            OUTPUT_DIM=3,  # R
        ),
        path = __INCLUDE_PATH__+'/maps/scene_radiance.h'
    )
    
    def __init__(self,
                 surfaces: Geometry,
                 environment: MapBase,
                 surface_emission: Optional[List[MapBase]] = None,
                 surface_scattering_sampler: Optional[List[MapBase]] = None,
                 inside_media: List[MapBase] = None,
                 outside_media: List[MapBase] = None,
                 media_filter: Literal['none', 'transmitted', 'emitted', 'scattered'] = 'none'
                 ):
        super().__init__(surfaces.get_number_of_patches(), MEDIA_FILTER = {'none': 0, 'transmitted': 1, 'emitted': 2, 'scattered': 3}[media_filter])
        self.surfaces = surfaces
        self.environment = environment
        self.sss_tensor = None if surface_scattering_sampler is None else torch.zeros(surfaces.get_number_of_patches(), dtype=torch.int64, device=device())
        self.se_tensor = None if surface_emission is None else torch.zeros(surfaces.get_number_of_patches(), dtype=torch.int64, device=device())
        self.im_tensor = None if inside_media is None else torch.zeros(surfaces.get_number_of_patches(), dtype=torch.int64, device=device())
        self.om_tensor = None if outside_media is None else torch.zeros(surfaces.get_number_of_patches(), dtype=torch.int64, device=device())
        if surface_scattering_sampler is not None:
            for i, sss in enumerate(surface_scattering_sampler):
                self.sss_tensor[i] = 0 if sss is None else sss.__bindable__.device_ptr
        if surface_emission is not None:
            for i, e in enumerate(surface_emission):
                self.se_tensor[i] = 0 if e is None else e.__bindable__.device_ptr
        if inside_media is not None:
            for i, mi in enumerate(inside_media):
                self.im_tensor[i] = 0 if mi is None else mi.__bindable__.device_ptr
        if outside_media is not None:
            for i, mo in enumerate(outside_media):
                self.om_tensor[i] = 0 if mo is None else mo.__bindable__.device_ptr
        self.surface_scattering_sampler = self.sss_tensor
        self.surface_emission = self.se_tensor
        self.inside_media = self.im_tensor
        self.outside_media = self.om_tensor
        modules = torch.nn.ModuleList()
        if surface_emission is not None:
            modules.extend(surface_emission)
        if surface_scattering_sampler is not None:
            modules.extend(surface_scattering_sampler)
        if inside_media is not None:
            modules.extend(inside_media)
        if outside_media is not None:
            modules.extend(outside_media)
        self.used_modules = modules


# def create_emitters(
#         surfaces: MapBase,
#         media_filter: Literal['none', 'transmittance', 'emitted', 'scattered'] = 'transmittance',
#         environment: Optional[MapBase] = None,
#         surface_emission: Optional[List[MapBase]] = None,
#         inside_media: Optional[List[MapBase]] = None,
#         outside_media: Optional[List[MapBase]] = None
# )

# class Scene(MapBase):
#     pass
