#include "stdafx.h"
#include <array>
#include <chrono>
#include <mutex>
#include <thread>

__forceinline unsigned int morton(unsigned int x) {
	x = (x ^ (x << 16)) & 0xff0000ff, x = (x ^ (x << 8)) & 0x0300f00f;
	x = (x ^ (x << 4)) & 0x030c30c3, x = (x ^ (x << 2)) & 0x09249249;
	return x;
}

//std::mutex mtx;

void Scene::generate_supercell(int start_x, int start_y, int start_z) {
	SimplexNoise noise(1.f, 1.f, 2.f, 0.5f);

	std::vector<float> heights;
	heights.reserve(supergrid_cell_size * brick_size * supergrid_cell_size * brick_size);

	for (int y = 0; y < supergrid_cell_size * brick_size; y++) {
		for (int x = 0; x < supergrid_cell_size * brick_size; x++) {
			float h = noise.fractal(7, (start_x * supergrid_cell_size * brick_size + x) / 1024.f, (start_y * supergrid_cell_size * brick_size + y) / 1024.f) * (grid_height / 2.f) + (grid_height / 2.f);
			heights.push_back(h);
		}
	}

	auto supercell = std::make_unique<Supercell>();

	supercell->indices.resize(supergrid_cell_size * supergrid_cell_size * supergrid_cell_size, 0);
	for (int z = 0; z < supergrid_cell_size; z++) {
		for (int y = 0; y < supergrid_cell_size; y++) {
			for (int x = 0; x < supergrid_cell_size; x++) {
				Brick brick{};
				for (int cell_x = 0; cell_x < brick_size; cell_x++) {
					for (int cell_y = 0; cell_y < brick_size; cell_y++) {
						for (int cell_z = 0; cell_z < brick_size; cell_z++) {
							if ((start_z * supergrid_cell_size + z) * brick_size + cell_z < heights[cell_x + x * brick_size + (cell_y + y * brick_size) * brick_size * supergrid_cell_size]) {
								uint32_t sub_data = (cell_x + cell_y * brick_size + cell_z * brick_size * brick_size) / (sizeof(uint32_t) * 8);
								uint32_t bit_position = (cell_x + cell_y * brick_size + cell_z * brick_size * brick_size) % (sizeof(uint32_t) * 8);
								brick.data[sub_data] |= (1 << bit_position);
							}
						}
					}
				}
				bool empty = true;
				for (int j = 0; j < cell_members; j++) {
					empty = empty && !brick.data[j];
				}
				if (!empty) {
					supercell->bricks.push_back(brick);
					// ToDo morton order
					supercell->indices[x + y * supergrid_cell_size + z * supergrid_cell_size * supergrid_cell_size] = (supercell->bricks.size() - 1) | brick_loaded_bit;
				}
			}
		}
	}
	supergrid[start_x + start_y * supergrid_xy + start_z * supergrid_xy * supergrid_xy] = std::move(supercell);
}

void Scene::generate() {
	auto begin = std::chrono::steady_clock::now();

	supergrid.resize(supergrid_z * supergrid_xy * supergrid_xy);
	for (int z = 0; z < supergrid_z; z++) {
		for (int y = 0; y < supergrid_xy; y++) {
			for (int x = 0; x < supergrid_xy; x++) {
				generate_supercell(x, y, z);
			}
		}
	}

	std::cout << "Generation took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();

	// Copy supercell content to GPU
	std::vector<Brick*> gpu_supercells;
	std::vector<uint16_t*> gpu_superindices;
	std::vector<uint16_t> temp_indices(supergrid_cell_size * supergrid_cell_size * supergrid_cell_size);

	for (auto& i : supergrid) {
		for (int j = 0; j < i->indices.size(); j++) {
			if (i->indices[j] & brick_loaded_bit) {
				temp_indices[j] = brick_unloaded_bit;
			} else {
				temp_indices[j] = 0;
			}
		}

		gpu_supercells.push_back(nullptr);
		gpu_superindices.push_back(nullptr);
		// cudaMallocPitch for proper row alignment?
		cuda(Malloc(&gpu_supercells.back(), 256 * sizeof(Brick)));
		//cuda(Malloc(&gpu_supercells.back(), i->bricks.size() * sizeof(Brick)));
		//cuda(Memcpy(gpu_supercells.back(), i->bricks.data(), i->bricks.size() * sizeof(Brick), cudaMemcpyKind::cudaMemcpyHostToDevice));
		cuda(Malloc(&gpu_superindices.back(), i->indices.size() * sizeof(uint16_t)));

		cuda(Memcpy(gpu_superindices.back(), temp_indices.data(), temp_indices.size() * sizeof(uint16_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
		i->gpu_brick_location = gpu_supercells.back();
		i->gpu_indices_location = gpu_superindices.back();
	}

	// Copy pointers for supergrid to GPU
	cuda(Malloc(&gpuScene.bricks, gpu_supercells.size() * sizeof(Brick*)));
	cuda(Memcpy(gpuScene.bricks, gpu_supercells.data(), gpu_supercells.size() * sizeof(Brick*), cudaMemcpyKind::cudaMemcpyHostToDevice));
	cuda(Malloc(&gpuScene.indices, gpu_superindices.size() * sizeof(uint16_t*)));
	cuda(Memcpy(gpuScene.indices, gpu_superindices.data(), gpu_superindices.size() * sizeof(uint16_t*), cudaMemcpyKind::cudaMemcpyHostToDevice));

	cuda(Malloc(&gpuScene.brick_load_queue_count, 4));
	cuda(Memset(gpuScene.brick_load_queue_count, 0, 4));

	cuda(Malloc(&gpuScene.brick_load_queue, brick_load_queue_size * sizeof(glm::vec3)));
	cuda(Memset(gpuScene.brick_load_queue, 0, brick_load_queue_size * sizeof(glm::vec3)));

	std::cout << "Allocation took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();
}

void Scene::process_load_queue() {
	uint32_t brick_to_load_count = 0;
	cudaMemcpy(&brick_to_load_count, gpuScene.brick_load_queue_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	//std::vector<glm::ivec3> bricks_to_load;
	//bricks_to_load.resize(brick_load_queue_size);

	//cudaMemcpy(bricks_to_load.data(), gpuScene.brick_load_queue, brick_load_queue_size * sizeof(glm::ivec3), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < brick_to_load_count; i++) {
	//	const glm::ivec3& pos = bricks_to_load[i];
	////for (const auto& i : bricks_to_load) {
	//	int supergrid_index = pos.x / supergrid_cell_size + pos.y / supergrid_cell_size * supergrid_xy + pos.z / supergrid_cell_size * supergrid_xy * supergrid_xy;
	//	auto& t = supergrid[supergrid_index];

	//	if (t->gpu_index_highest + 1 == t->gpu_count) {
	//		// have to relocate the supercell storage

	//		Brick* previous = t->gpu_brick_location;
	//		cuda(Malloc(&t->gpu_brick_location, t->gpu_count * 2 * sizeof(Brick)));
	//		cudaMemcpy(t->gpu_brick_location, previous, t->gpu_count * sizeof(Brick), cudaMemcpyDeviceToDevice);
	//		cudaMemcpy(gpuScene.bricks + supergrid_index, t->gpu_brick_location, sizeof(Brick*), cudaMemcpyHostToDevice);
	//		t->gpu_count *= 2;
	//	}

	//	glm::ivec3 block_pos = pos % supergrid_cell_size;
	//	
	//	uint16_t index = t->indices[block_pos.x + block_pos.y * supergrid_cell_size + block_pos.z * supergrid_cell_size * supergrid_cell_size];

	//	if (index == 0) {
	//		// empty do nothing
	//		assert(false);
	//	}

	//	Brick& to_upload = t->bricks[index];
	//	
	//	uint16_t new_index = (t->gpu_index_highest++) | brick_loaded_bit;
	//	t->gpu_indices[index] = new_index;
	//	// upload to_upload to the correct location


	//	cuda(Memcpy(t->gpu_indices_location, &new_index, sizeof(uint16_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
	//	cuda(Memcpy(t->gpu_brick_location, &to_upload, sizeof(Brick), cudaMemcpyKind::cudaMemcpyHostToDevice));
	//}

	std::cout << brick_to_load_count << "\n";
}