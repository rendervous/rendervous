# Rendervous - Rendering and Learning and vice versa


Rendervous is a project designed primarily for academic purposes. The core idea is the integration of GPU rendering capabilities including ray-tracing (Vulkan) with deep learning (Pytorch).

As a result you have a differentiable renderer that can be included in learning models and vice versa, learning models that can be included as renderer components (materials, scattering functions, parameters, etc).


## Maps

The main concept behind rendervous is a *map* derived from `MapBase`. 

### Implicit parameters

Maps represents functions with explicit inputs and implicit parameters $f(x;\theta) : R^{N+P} --> R^{M}$.
In here, $x$ is the input of the function dependent implicitly on $\theta$. We will express all dependencies to parameterswith $\theta$ in this documentation. 

### Stochasticity

Another important feature of rendervous *maps* is the ability for stochastic evaluation. We will assume there is a hidden non-differentiable dependency of
all maps to a value in a unitary infinity-dimensional space $U$. This means that a map can be seen as a random variable, although the same value
can be re-evaluated any time, assuming we replay the same seed, equivalent to reevaluate all ''random'' variables in the same 
primal point $u$:

$$
X = x(u_0), Y = y(u_1), ...
$$

### Differentiability

Maps represents differentiable modules of `torch`, meaning they can 
backpropagate gradients with respect to an input. 
Given that the input also considers the primal space, 
the backward method needs to ''match'' in the way that 
they use the same seed sequence. We will explain this 
with more detail in another section.

### Sesors

Sensors represents functions in the form:

$$
I_k(\theta) = \int_{x \in \mathcal{D}} W_e(x) f(x) dx
$$







