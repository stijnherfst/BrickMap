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
	cudaResourceDesc viewCudaArrayResourceDesc;
	cudaSurfaceObject_t surf;

	cuda_interop();
	~cuda_interop();

	cudaError set_size(const int width, const int height);
	void blit();
};