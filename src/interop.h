#pragma once

#include "Scene.h"
#include <cuda_runtime.h>

struct cuda_interop {
	int width;
	int height;

	// GL buffers
	GLuint fb;
	GLuint rb;

	// CUDA resources
	cudaGraphicsResource* cgr = nullptr;
	cudaArray* ca = nullptr;

	cuda_interop();
	~cuda_interop();

	cudaError set_size(const int width, const int height);
	void blit();
};

cudaError launch_kernels(cudaArray_const_t array, glm::vec4* blit_buffer, Scene::GPUScene gpuScene, RayQueue* queue, RayQueue* queue2, ShadowQueue* shadowQueue);