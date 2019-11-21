#pragma once

class Scene {
public:
	struct GPUScene {
		uint8_t* voxels;
	} gpuScene;

	void generate();
};

struct Brick {
	uint32_t data[16];
};

class BrickGrid {
	uint8_t* brick_grid;
};