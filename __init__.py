rdv_device = 0
# command = "nvidia-smi --query-gpu=memory.used --format=csv"
# mems = sp.check_output(command.split()).decode('ascii').split('\n')[1:-1]
# for i, m in enumerate(mems):
#     if i not in [1]:
#         if int(m.split()[0]) < 20:
#             rdv_device = i
#             break
# assert rdv_device>=0, "All devices where in use!"

import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(rdv_device)

from rendervous.rendering._gmath import *
from rendervous.rendering import create_device as _create_device, external_sync, window, Format, execute_loop, \
    allow_cross_threading, GPUPtr, tensor, tensor_like

from ._internal import device

_create_device(device=rdv_device, debug=True)

from ._functions import *
from ._objects import *
from ._maps import start_engine, torch_fallback, Grid3DSensor, Box3DSensor, parameter, DummyExample,\
    grid2d, grid3d, xr_projection, AbsorptionOnlyVolume, TransmittanceDDA, normalized_box, \
    transmittance, constant, ray_direction, ray_position, ray_to_segment, TotalVariation, \
    Identity, InputSelectMap, \
    ray_box_intersection, DTCollisionIntegrator, SH_PDF, Grid3DTransmittanceRayIntegral, Grid3DTransmittanceDDARayIntegral
from ._scenes import *

start_engine()

