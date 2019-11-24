#pragma once

struct Brick {
	uint32_t data[16];
};

class BrickGrid {
	Brick* brick_grid;
};

class Scene {
public:
	struct GPUScene {
		uint8_t* voxels;
		BrickGrid* grid;
	} gpuScene;

	void generate();
};
