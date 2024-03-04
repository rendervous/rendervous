void medium_path_integrator(map_object, vec3 x, vec3 w, float d, out float T, out vec3 xo, out vec3 wo, out vec3 W, out vec3 A);
void medium_path_integrator_bw(map_object, vec3 x, vec3 w, float d, float outT, vec3 outW, vec3 outA, float dL_dT, vec3 dL_dW, vec3 dL_dA);

vec3 gathering(map_object, vec3 xs, vec3 ws)
{
    float _output[3];
    forward(parameters.gathering, float[6](xs.x, xs.y, xs.z, ws.x, ws.y, ws.z), _output);
    return vec3(_output[0], _output[1], _output[2]);
}

void gathering_bw(map_object, vec3 xs, vec3 ws, vec3 dL_dA)
{
    backward(parameters.gathering, float[6](xs.x, xs.y, xs.z, ws.x, ws.y, ws.z), float[3](dL_dA.x, dL_dA.y, dL_dA.z));
}

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float T;
    vec3 xo, wo;
    vec3 W, A;
    BRANCH_SEED(before_path)
    medium_path_integrator(object, x, w, d, T, xo, wo, W, A);
    SET_SEED(before_path)
    _output = float[13](
        T,
        xo.x, xo.y, xo.z,
        wo.x, wo.y, wo.z,
        W.x, W.y, W.z,
        A.x, A.y, A.z
    );
}

BACKWARD_USING_OUTPUT {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float out_T = _output[0];
    vec3 out_W = vec3(_output[7], _output[8], _output[9]);
    vec3 out_A = vec3(_output[10], _output[11], _output[12]);
    float dL_dT = _output_grad[0];
    vec3 dL_dW = vec3(_output_grad[7], _output_grad[8], _output_grad[9]);
    vec3 dL_dA = vec3(_output_grad[10], _output_grad[11], _output_grad[12]);
    BRANCH_SEED(before_path)
    // replay
    medium_path_integrator_bw(
        object, x, w, d, out_T, out_W, out_A, dL_dT, dL_dW, dL_dA
    );
    SET_SEED(before_path)
    // TODO: no input update here! check how can be done...
}
