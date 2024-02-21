

void environment(map_object, vec3 x, vec3 we, out vec3 E)
{
    float[3] _output;// = float[3](0.0, 0.0, 0.0);
    forward(parameters.environment, float[6](x.x, x.y, x.z, we.x, we.y, we.z), _output);
    E = vec3(_output[0], _output[1], _output[2]);
}