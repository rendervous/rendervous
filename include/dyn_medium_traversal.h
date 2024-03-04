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


void medium_traversal_bw(
    map_object, GPUPtr medium_ptr,
    vec3 x, vec3 w, float d, // input ray segment
    float T, vec3 W, vec3 A, // cached output
    float dL_dT, vec3 dL_dW, vec3 dL_dA // output gradients
)
{
    float _input[7] = float[7](x.x, x.y, x.z, w.x, w.y, w.z, d);
    float _output[13] = float[13](T, 0, 0, 0, 0, 0, 0, W.x, W.y, W.z, A.x, A.y, A.z);
    float _output_grad[13] = float[13](dL_dT, 0, 0, 0, 0, 0, 0, dL_dW.x, dL_dW.y, dL_dW.z, dL_dA.x, dL_dA.y, dL_dA.z);
    float _input_grad[7];
    dynamic_backward(object, medium_ptr, _input, _output, _output_grad, _input_grad);
}