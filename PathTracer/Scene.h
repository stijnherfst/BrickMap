#pragma once

struct Brick {
	uint32_t data[cell_members];
};

class Scene {
public:
	struct GPUScene {
		uint32_t* brick_grid;
		Brick* bricks;
	} gpuScene;

	void generate();
};
