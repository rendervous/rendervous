#include "sc_boundaries.h"
#include "dyn_medium_traversal.h"
#include "sc_scene_visibility.h"


FORWARD {
    vec3 xa = vec3(_input[0], _input[1], _input[2]);
    vec3 xb = vec3(_input[3], _input[4], _input[5]);
    float T = scene_visibility(object, xa, xb);
    _output = float[1](T);
}

BACKWARD_USING_OUTPUT {
    vec3 xa = vec3(_input[0], _input[1], _input[2]);
    vec3 xb = vec3(_input[3], _input[4], _input[5]);
    float T = _output[0];
    float dL_dT = _output_grad[0];
    scene_visibility_bw(object, xa, xb, T, dL_dT);

    // TODO: NO INPUT GRAD!
}
