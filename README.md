# Rendervous - Rendering and Learning and vice versa

Rendervous is a project designed primarily for academic purposes. The core idea is the integration of GPU rendering capabilities including ray-tracing (Vulkan) with deep learning (Pytorch).

As a result you have a differentiable renderer that can be included in learning models and vice versa, learning models that can be included as renderer components (materials, scattering functions, parameters, etc).




## Dependencies

- torch
- cffi
- pywin32 (Windows)
- cuda-python (if cuda device can be used)

### Secondary dependencies
- matplotlib (most of the offline-rendering examples)

### Interactive examples 
- imgui
- glfw (for interactive examples)

