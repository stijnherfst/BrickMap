#include "stdafx.h"

cuda_interop::cuda_interop() {
	glCreateRenderbuffers(1, &rb);
	glCreateFramebuffers(1, &fb);

	// attach rbo to fbo
	glNamedFramebufferRenderbuffer(fb, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
}

cuda_interop::~cuda_interop() {
	cudaError cuda_err;

	// unregister CUDA resources
	if (cgr != nullptr)
		cuda_err = cuda(GraphicsUnregisterResource(cgr));

	glDeleteRenderbuffers(1, &rb);
	glDeleteFramebuffers(1, &fb);
}

cudaError cuda_interop::set_size(const int width, const int height) {
	cudaError cuda_err = cudaSuccess;

	this->width = width;
	this->height = height;

	// unregister resource
	if (cgr != nullptr)
		cuda_err = cuda(GraphicsUnregisterResource(cgr));

	// resize rbo
	glNamedRenderbufferStorage(rb, GL_RGBA32F, width, height);

	// register rbo
	cuda_err = cuda(GraphicsGLRegisterImage(&cgr, rb, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard));

	// map graphics resources
	cuda_err = cuda(GraphicsMapResources(1, &cgr, 0));

	// get CUDA Array refernces
	cuda_err = cuda(GraphicsSubResourceGetMappedArray(&ca, cgr, 0, 0));

	memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
	viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
	viewCudaArrayResourceDesc.res.array.array = ca;

	cuda(CreateSurfaceObject(&surf, &viewCudaArrayResourceDesc));

	// unmap graphics resources
	cuda_err = cuda(GraphicsUnmapResources(1, &cgr, 0));

	return cuda_err;
}

void cuda_interop::blit() {
	glBlitNamedFramebuffer(fb, 0, 0, 0, width, height, 0, height, width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}