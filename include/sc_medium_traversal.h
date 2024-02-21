/*
Represents objects that can traverse volume scattering radiance
*/

void medium_traversal(map_object, GPUPtr medium_ptr,
    vec3 x, vec3 w, float d, // input ray segment
    out float T, out vec3 xe, out vec3 we, out vec3 W, out vec3 A
    )
{
    float _input[7] = float[7](x.x, x.y, x.z, w.x, w.y, w.z, d);
    float _output[13];
    dynamic_forward(object, medium_ptr, _input, _output);
    T = _output[0];
    xe = vec3(_output[1], _output[2], _output[3]);
    we = vec3(_output[4], _output[5], _output[6]);
    W = vec3(_output[7], _output[8], _output[9]);
    A = vec3(_output[10], _output[11], _output[12]);
}