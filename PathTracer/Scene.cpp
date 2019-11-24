#include "stdafx.h"
#include <array>

void Scene::generate() {
	{
		std::vector<uint8_t> scene;
		scene.resize(grid_size * grid_size * grid_size);

		for (int x = 0; x < grid_size; x++) {
			for (int y = 0; y < grid_size; y++) {
				for (int z = 0; z < grid_size; z++) {
					scene[x + y * grid_size + z * grid_size * grid_size] = sqrt((x - 128) * (x - 128) + (y - 128) * (y - 128) + (z - 128) * (z - 128)) < 128;
				}
			}
		}

		cuda(Malloc(&gpuScene.voxels, grid_size * grid_size * grid_size));
		cuda(Memcpy(&gpuScene.voxels[0], &scene[0], grid_size * grid_size * grid_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	}

	/*{
		cuda(Malloc(&gpuScene.grid, cells * cells * cells));

		std::vector<Brick> scene;
		scene.resize(cells * cells * cells);

		for (int x = 0; x < grid_size; x++) {
			for (int y = 0; y < grid_size; y++) {
				for (int z = 0; z < grid_size; z++) {
					uint32_t sub_x = x % cell_size;
					uint32_t sub_y = (y % cell_size) * cell_size;
					uint32_t sub_z = (z % cell_size) * cell_size * cell_size;

					uint32_t sub_data = (sub_x + sub_y + sub_z) / sizeof(uint32_t);
					uint32_t bit_position = (sub_x + sub_y + sub_z) % (sizeof(uint32_t) * 8);

					bool set = sqrt((x - 128) * (x - 128) + (y - 128) * (y - 128) + (z - 128) * (z - 128)) < 128;

					if (set) {
						scene[sub_x + sub_y + sub_z].data[sub_data] |= (1 << bit_position);
					}
				}
			}
		}
	}*/
}
