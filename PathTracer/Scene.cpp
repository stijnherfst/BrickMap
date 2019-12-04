#include "stdafx.h"
#include <array>
#include <chrono>

void Scene::generate() {
	std::vector<Brick> scene;
	scene.resize(cells * cells * cells_height);
		
	auto begin = std::chrono::steady_clock::now();

	for (int y = 0; y < grid_size; y++) {
		for (int x = 0; x < grid_size; x++) {
			float height = SimplexNoise::noise(x / 256.f, y / 256.f) * (grid_height / 2.f) + (grid_height / 2.f);

			for (int z = 0; z < height; z++) {
				uint32_t sub_x = x / cell_size;
				uint32_t sub_y = (y / cell_size) * cells;
				uint32_t sub_z = (z / cell_size) * cells * cells;

				uint32_t cell_x = x % cell_size;
				uint32_t cell_y = (y % cell_size) * cell_size;
				uint32_t cell_z = (z % cell_size) * cell_size * cell_size;

				uint32_t sub_data = (cell_x + cell_y + cell_z) / (sizeof(uint32_t) * 8);
				uint32_t bit_position = (cell_x + cell_y + cell_z) % (sizeof(uint32_t) * 8);

				scene[sub_x + sub_y + sub_z].data[sub_data] |= (1 << bit_position);
			}
		}
	}

	std::cout << "Generation took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();

	std::vector<Brick*> grid;
	grid.resize(cells * cells * cells, nullptr);

	int count = 0;
	for (int i = 0; i < cells * cells * cells_height; i++) {
		bool empty = true;
		for (int j = 0; j < cell_members; j++) {
			empty = empty && !scene[i].data[j];
		}
		if (!empty) {
			grid[i] = (Brick*)1;
			count++;
		}
	}

	std::cout << "Counting bricks took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();

	Brick* gpu_brick_pointers;
	cuda(Malloc(&gpu_brick_pointers, count * sizeof(Brick)));

	int index = 0;
	for (int i = 0; i < cells * cells * cells_height; i++) {
		if (grid[i] == (Brick*)1) {
			grid[i] = gpu_brick_pointers + index;

			cuda(Memcpy(grid[i], &scene[i], sizeof(Brick), cudaMemcpyKind::cudaMemcpyHostToDevice));
			index++;
		}
	}
	cuda(Malloc(&gpuScene.brick_grid, cells * cells * cells * sizeof(Brick*)));
	cuda(Memcpy(gpuScene.brick_grid, &grid[0], cells * cells * cells * sizeof(Brick*), cudaMemcpyKind::cudaMemcpyHostToDevice));

	std::cout << "Allocating/Copying took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
}
