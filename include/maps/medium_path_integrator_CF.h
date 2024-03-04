// This code is an implementation of a medium path integrator
// Close-form integrator for homogeneous cases
#include "medium_path_integrator_interface.h"

#include "sc_phase_sampler.h"

void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out float T, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A)
{
    float sigma = param_float(parameters.sigma);
    T = exp(-d * sigma);
    W = vec3(0.0);
    A = vec3(0.0);
    float t = -log(1 - random()*(1 - T)) / max(0.0000001, sigma);
    //t = min(t, d);
    xo = x + w * t;
    #if MEDIUM_FILTER & 1
    vec3 albedo = param_vec3(parameters.scattering_albedo);
    float weight, pdf;
    wo = sample_phase(object, w, weight, pdf);
    W = albedo * weight;
    #endif
    #if MEDIUM_FILTER & 2
    A = gathering(object, xo, w);
    #endif
}

void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, float outT, vec3 outW, vec3 outA, float dL_dT, vec3 dL_dW, vec3 dL_dA)
{
}