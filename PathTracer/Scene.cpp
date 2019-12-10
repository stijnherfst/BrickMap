#include "stdafx.h"
#include <array>
#include <chrono>
#include <thread>

__forceinline unsigned int morton(unsigned int x) {
	x = (x ^ (x << 16)) & 0xff0000ff, x = (x ^ (x << 8)) & 0x0300f00f;
	x = (x ^ (x << 4)) & 0x030c30c3, x = (x ^ (x << 2)) & 0x09249249;
	return x;
}

void Scene::generate() {
	std::vector<Brick> scene;
	scene.resize(cells * cells * cells_height);

	auto begin = std::chrono::steady_clock::now();

	auto func = [&](int start, int end) {
		//FastNoiseSIMD* noise = FastNoiseSIMD::NewFastNoiseSIMD();

		//float* noiseSet = noise->GetSimplexSet(0, start * cell_size, 0, cells * cell_size, (end - start) * cell_size, 1);  
		SimplexNoise noise(1.f, 1.f, 2.f, 0.5f);
		
		for (int y = start; y < end; y++) {
			for (int x = 0; x < cells; x++) {
				for (int cell_x = 0; cell_x < cell_size; cell_x++) {
					for (int cell_y = 0; cell_y < cell_size; cell_y++) {
						//float height = SimplexNoise::noise((x * cell_size + cell_x) / 256.f, (y * cell_size + cell_y) / 256.f); // * (grid_height / 2.f) + (grid_height / 2.f);
						//float height = noiseSet[x * cell_size + cell_x + ((y - start) * cell_size + cell_y) * ((end - start) * cell_size)]; //*(grid_height / 2.f) + (grid_height / 2.f);
						float height = noise.fractal(7, (x * cell_size + cell_x) / 1024.f, (y * cell_size + cell_y) / 1024.f) * (grid_height / 2.f) + (grid_height / 2.f);

						for (int z = 0; z < height; z++) {
							uint32_t sub_data = (cell_x + cell_y * cell_size + (z % cell_size) * cell_size * cell_size) / (sizeof(uint32_t) * 8);
							uint32_t bit_position = (cell_x + cell_y * cell_size + (z % cell_size) * cell_size * cell_size) % (sizeof(uint32_t) * 8);


							//scene[x + y * cells + (z / cell_size) * cells * cells].data[sub_data] |= (1 << bit_position);
							scene[x + y * cells + (z / cell_size) * cells * cells].data[sub_data] |= (1 << bit_position);
						}
					}
				}
			}
		}
	//	FastNoiseSIMD::FreeNoiseSet(noiseSet);
	};


	constexpr int thread_count = 16;
	std::thread threads[thread_count];
	for (int i = 0; i < thread_count; i++) {
		threads[i] = std::thread(func, (cells / thread_count) * i, (cells / thread_count) * (i + 1));
	}

	for (int i = 0; i < thread_count; i++) {
		threads[i].join();
	}

	std::cout << "Generation took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();

	std::vector<uint32_t> grid;
	grid.resize(cells * cells * cells_height, UINT_MAX);

	int count = 0;
	for (int i = 0; i < cells * cells * cells_height; i++) {
		for (int j = 0; j < cell_members; j++) {
			if (scene[i].data[j]) {
				int mort = morton(i % cells) + (morton(i / cells % cells) << 1) + (morton(i / (cells_height * cells_height)) << 2);
				grid[mort] = count;
				count++;
				break;
			}
		}
	}

	std::cout << "Counting bricks took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();

	// Allocate space for grid containing indices into the brick storage
	cuda(Malloc(&gpuScene.brick_grid, cells * cells * cells_height * sizeof(uint32_t)));
	cuda(Memcpy(gpuScene.brick_grid, grid.data(), cells * cells * cells_height * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyHostToDevice));

	// Compact grid
	auto new_end = std::remove_if(scene.begin(), scene.end(), [](Brick& brick) {
		for (int j = 0; j < cell_members; j++) {
			if (brick.data[j]) {
				return false;
			}
		}
		return true;
	});
	scene.erase(new_end, scene.end());

	// Allocate space and copy bricks
	cuda(Malloc(&gpuScene.bricks, count * sizeof(Brick)));
	cuda(Memcpy(gpuScene.bricks, scene.data(), count * sizeof(Brick), cudaMemcpyKind::cudaMemcpyHostToDevice));

	std::cout << "Allocating/Copying took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
}
