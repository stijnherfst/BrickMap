#pragma once

#include <cuda_runtime.h>
#include <state.h>

cudaError launch_kernels(State& state, cudaSurfaceObject_t surf, glm::vec4* blit_buffer, Scene::GPUScene gpuScene, RayQueue* queue, RayQueue* queue2, ShadowQueue* shadowQueue);