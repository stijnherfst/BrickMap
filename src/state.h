#pragma once

#include "stdafx.h"

struct State {
	cuda_interop interop;

	RayQueue* ray_buffer_work;
	RayQueue* ray_buffer_next;
	ShadowQueue* shadow_queue_buffer;
	glm::vec4* blit_buffer;

	size_t screen_width;
	size_t screen_height;

	State(size_t screen_width, size_t screen_height)
		: screen_width(screen_width)
		, screen_height(screen_height) {
		cuda(Malloc(&ray_buffer_work, ray_queue_buffer_size * sizeof(RayQueue)));
		cuda(Malloc(&ray_buffer_next, ray_queue_buffer_size * sizeof(RayQueue)));
		cuda(Malloc(&shadow_queue_buffer, ray_queue_buffer_size * sizeof(ShadowQueue)));
		cuda(Malloc(&blit_buffer, screen_width * screen_height * sizeof(glm::vec4)));
		interop.set_size(screen_width, screen_height);
	}

	void screen_resize(size_t screen_width, size_t screen_height) {
		this->screen_width = screen_width;
		this->screen_height = screen_height;

		cuda(Free(blit_buffer));
		cuda(Malloc(&blit_buffer, screen_width * screen_height * sizeof(glm::vec4)));
		interop.set_size(screen_width, screen_height);
	}
};
