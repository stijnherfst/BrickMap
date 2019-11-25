#pragma once

struct Brick {
	uint32_t data[cell_members];
};

class Scene {
public:
	struct GPUScene {
		uint8_t* voxels;
		Brick** grid;
	} gpuScene;

	void generate();
};
