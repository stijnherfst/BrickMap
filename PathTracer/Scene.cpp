#include "stdafx.h"
#include <array>
#include <chrono>
#include <thread>
void Scene::generate() {
	std::vector<Brick> scene;
	scene.resize(cells * cells * cells_height);

	auto begin = std::chrono::steady_clock::now();

	//std::for_each(std::execution::par_unseq)
	//for (int y = 0; y < grid_size; y++) {
	//	for (int x = 0; x < grid_size; x++) {
	//		float height = SimplexNoise::noise(x / 256.f, y / 256.f) * (grid_height / 2.f) + (grid_height / 2.f);

	//		for (int z = 0; z < height; z++) {
	//			uint32_t sub_x = x / cell_size;
	//			uint32_t sub_y = (y / cell_size) * cells;
	//			uint32_t sub_z = (z / cell_size) * cells * cells;

	//			uint32_t cell_x = x % cell_size;
	//			uint32_t cell_y = (y % cell_size) * cell_size;
	//			uint32_t cell_z = (z % cell_size) * cell_size * cell_size;

	//			uint32_t sub_data = (cell_x + cell_y + cell_z) / (sizeof(uint32_t) * 8);
	//			uint32_t bit_position = (cell_x + cell_y + cell_z) % (sizeof(uint32_t) * 8);

	//			scene[sub_x + sub_y + sub_z].data[sub_data] |= (1 << bit_position);
	//		}
	//	}
	//}

	//std::for_each(std::execution::par_unseq, )

	auto func = [&](int start, int end) {
		for (int y = start; y < end; y++) {
			for (int x = 0; x < cells; x++) {
				for (int cell_x = 0; cell_x < cell_size; cell_x++) {
					for (int cell_y = 0; cell_y < cell_size; cell_y++) {
						float height = SimplexNoise::noise((x * cell_size + cell_x) / 256.f, (y * cell_size + cell_y) / 256.f) * (grid_height / 2.f) + (grid_height / 2.f);

						for (int z = 0; z < height; z++) {
							/*uint32_t sub_x = x / cell_size;
						uint32_t sub_y = (y / cell_size) * cells;
						uint32_t sub_z = (z / cell_size) * cells * cells;*/

							/*uint32_t cell_x = x % cell_size;
						uint32_t cell_y = (y % cell_size) * cell_size;
						uint32_t cell_z = (z % cell_size) * cell_size * cell_size;*/

							uint32_t sub_data = (cell_x + cell_y * cell_size + (z % cell_size) * cell_size * cell_size) / (sizeof(uint32_t) * 8);

							//uint32_t sub_data = / (sizeof(uint32_t) * 8);
							uint32_t bit_position = (cell_x + cell_y * cell_size + (z % cell_size) * cell_size * cell_size) % (sizeof(uint32_t) * 8);

							scene[x + y * cells + (z / cell_size) * cells * cells].data[sub_data] |= (1 << bit_position);
						}
					}
				}
			}
		}
	};

	constexpr int thread_count = 16;
	std::thread threads[thread_count];
	for (int i = 0; i < thread_count; i++) {
		threads[i] = std::thread(func, (1024 / thread_count) * i, (1024 / thread_count) * (i + 1));
	}

	for (int i = 0; i < thread_count; i++) {
		threads[i].join();
	}
		//std::thread thread1(func, 0, 512);
	//std::thread thread2(func, 513, 1024);
	//thread1.join();
	//thread2.join();
	
	//std::thread thread2;

	//for (int y = 0; y < cells; y++) {
	//	for (int x = 0; x < cells; x++) {
	//		for (int cell_x = 0; cell_x < cell_size; cell_x++) {
	//			for (int cell_y = 0; cell_y < cell_size; cell_y++) {
	//				float height = SimplexNoise::noise((x * cell_size + cell_x) / 256.f, (y * cell_size + cell_y) / 256.f) * (grid_height / 2.f) + (grid_height / 2.f);

	//				for (int z = 0; z < height; z++) {
	//					/*uint32_t sub_x = x / cell_size;
	//					uint32_t sub_y = (y / cell_size) * cells;
	//					uint32_t sub_z = (z / cell_size) * cells * cells;*/

	//					/*uint32_t cell_x = x % cell_size;
	//					uint32_t cell_y = (y % cell_size) * cell_size;
	//					uint32_t cell_z = (z % cell_size) * cell_size * cell_size;*/

	//					uint32_t sub_data = (cell_x + cell_y * cell_size + (z % cell_size) * cell_size * cell_size) / (sizeof(uint32_t) * 8);



	//					//uint32_t sub_data = / (sizeof(uint32_t) * 8);
	//					uint32_t bit_position = (cell_x + cell_y * cell_size + (z % cell_size) * cell_size * cell_size) % (sizeof(uint32_t) * 8);

	//					scene[x + y * cells + (z / cell_size) * cells * cells].data[sub_data] |= (1 << bit_position);
	//				}
	//			}
	//		}
	//	}
	//}

	std::cout << "Generation took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();

	std::vector<uint32_t> grid;
	grid.resize(cells * cells * cells_height, UINT_MAX);

	int count = 0;
	for (int i = 0; i < cells * cells * cells_height; i++) {
		for (int j = 0; j < cell_members; j++) {
			if (scene[i].data[j]) {
				grid[i] = count;
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
