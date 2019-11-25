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
		cuda(Memcpy(gpuScene.voxels, &scene[0], grid_size * grid_size * grid_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	}

	{
		std::vector<Brick> scene;
		scene.resize(cells * cells * cells);

		for (int x = 0; x < grid_size; x++) {
			for (int y = 0; y < grid_size; y++) {
				for (int z = 0; z < grid_size; z++) {
					uint32_t sub_x = x / cell_size;
					uint32_t sub_y = (y / cell_size) * cells;
					uint32_t sub_z = (z / cell_size) * cells * cells;

					uint32_t cell_x = x % cell_size;
					uint32_t cell_y = (y % cell_size) * cell_size;
					uint32_t cell_z = (z % cell_size) * cell_size * cell_size;

					uint32_t sub_data = (cell_x + cell_y + cell_z) / (sizeof(uint32_t) * 8);
					uint32_t bit_position = (cell_x + cell_y + cell_z) % (sizeof(uint32_t) * 8);

					bool set = sqrt((x - 128) * (x - 128) + (y - 128) * (y - 128) + (z - 128) * (z - 128)) < 128;

					if (set) {
						scene[sub_x + sub_y + sub_z].data[sub_data] |= (1 << bit_position);
					}
				}
			}
		}


		cuda(Malloc(&gpuScene.grid, cells * cells * cells * sizeof(Brick*)));
		std::vector<Brick*> grid{};
		grid.resize(cells * cells * cells);

		for (int i = 0; i < cells * cells * cells; i++) {
			bool empty = true;
			for (int j = 0; j < cell_members; j++) {
				empty = empty && !scene[i].data[j];
			}
			if (!empty) {
				cuda(Malloc(&grid[i], sizeof(Brick)));
				cuda(Memcpy(grid[i], &scene[i], sizeof(Brick), cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
		}

		//int count = 0;
		//for (int i = 0; i < cells * cells * cells; i++) {
		//	if (grid[i] != nullptr) {
		//		count++;
		//	}
		//}
		//printf("%i", count);
		cuda(Memcpy(gpuScene.grid, &grid[0], cells * cells * cells * sizeof(Brick*), cudaMemcpyKind::cudaMemcpyHostToDevice));
	}
}
