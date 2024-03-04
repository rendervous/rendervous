#include "sc_environment.h"
#include "sc_surfaces.h"
#include "sc_boundaries.h"
#include "dyn_surface_scattering_sampler.h"
#include "dyn_surface_gathering.h"
#include "dyn_medium_traversal.h"
#include "sc_path_tracing.h"


FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 R = path_trace(object, x, w);
    _output = float[3](R.x, R.y, R.z);
}
