#pragma once

struct Brick {
	uint32_t data[cell_members];
};

class Scene {
public:
	struct GPUScene {
		Brick** brick_grid;
	} gpuScene;

	void generate();
};
