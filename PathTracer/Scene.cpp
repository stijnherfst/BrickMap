#include "stdafx.h"
#include <array>

void Scene::Load(const char path[]) {

	std::vector<uint8_t> scene;
	scene.resize(grid_size * grid_size * grid_size);

	for (int x = 0; x < grid_size; x++) {
		for (int y = 0; y < grid_size; y++) {
			for (int z = 0; z < grid_size; z++) {
				scene[x + y * grid_size + z * grid_size * grid_size] = sqrt(32 * 32 - x * x + y * y + z * z) < 128;
			}
		}
	}

	cuda(Malloc(&gpuScene.voxels, grid_size * grid_size * grid_size));
	cuda(Memcpy(&gpuScene.voxels[0], &scene[0], grid_size * grid_size * grid_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
}
