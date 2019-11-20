#pragma once

class Scene {
public:
	struct GPUScene {
		uint8_t* voxels;
	} gpuScene;

	void Load(const char path[]);
};