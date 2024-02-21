

float visibility(map_object, vec3 xa, vec3 xb)
{
    float _output[1];
    forward(parameters.visibility, float[6](xa.x, xa.y, xa.z, xb.x, xb.y, xb.z), _output);
    return _output[0];
}

float ray_visibility(map_object, vec3 x, vec3 w)
{
    vec3 xb = w * 10000 + x;
    vec3 xa = x;
    float _output[1];
    forward(parameters.visibility, float[6](xa.x, xa.y, xa.z, xb.x, xb.y, xb.z), _output);
    return _output[0];
}