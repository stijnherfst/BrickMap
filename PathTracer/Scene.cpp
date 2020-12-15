#include "stdafx.h"
#include <array>
#include <chrono>
#include <thread>

cudaStream_t load_stream;


//uint64_t xy_to_morton(uint32_t x, uint32_t y) {
//	return _pdep_u32(x, 0x55555555) | _pdep_u32(y, 0xaaaaaaaa);
//}

__forceinline uint32_t morton(uint32_t x) {
	x = (x ^ (x << 16)) & 0xff0000ff;
	x = (x ^ (x << 8)) & 0x0300f00f;
	x = (x ^ (x << 4)) & 0x030c30c3;
	x = (x ^ (x << 2)) & 0x09249249;
	return x;
}

uint64_t morton3(uint32_t x, uint32_t y, uint32_t z) {
	return morton(x) | (morton(y) << 1ull) | (morton(z) << 2ull);
}

uint32_t MortonEncode3(uint32_t x, uint32_t y, uint32_t z) {
	return _pdep_u32(z, 0x24924924) | _pdep_u32(y, 0x12492492) | _pdep_u32(x, 0x09249249);
}

Scene::Scene() {
	cuda(HostAlloc(&bricks_to_load, brick_load_queue_size * sizeof(glm::ivec3), cudaHostAllocDefault));
	cuda(HostAlloc(&brick_gpu_staging, brick_load_queue_size * sizeof(Brick), cudaHostAllocDefault));
	cuda(HostAlloc(&indices_gpu_staging, brick_load_queue_size * sizeof(uint32_t), cudaHostAllocDefault));

	cudaStreamCreate(&load_stream);
	cudaStreamCreate(&kernel_stream);
}

Scene::~Scene() {
	cudaFreeHost(&bricks_to_load);
	cudaFreeHost(&brick_gpu_staging);
	cudaFreeHost(&indices_gpu_staging);
}

void Scene::generate_supercell(int start_x, int start_y, int start_z) {
	SimplexNoise noise(1.f, 1.f, 2.f, 0.5f);

	std::vector<float> heights;
	heights.reserve(supergrid_cell_size * brick_size * supergrid_cell_size * brick_size);

	for (int y = 0; y < supergrid_cell_size * brick_size; y++) {
		for (int x = 0; x < supergrid_cell_size * brick_size; x++) {
			float h = 1 - std::abs(noise.fractal(7, (start_x * supergrid_cell_size * brick_size + x) / 1024.f, (start_y * supergrid_cell_size * brick_size + y) / 1024.f));
			//float h = noise.fractal(7, (start_x * supergrid_cell_size * brick_size + x) / 1024.f, (start_y * supergrid_cell_size * brick_size + y) / 1024.f);
			h *= grid_height;//	/ 2.f; 
			//h += grid_height / 2.f;
			heights.push_back(h);
		}
	}
	
	//FastNoiseSIMD* myNoise = FastNoiseSIMD::NewFastNoiseSIMD();
	////myNoise->set
	//myNoise->SetFrequency(0.005f);
	////myNoise->setampl(1.f);
	////myNoise->SetFrequency(1.f);
	////myNoise->setac(1.f);
	//myNoise->SetFractalOctaves(7);
	//float* noiseSet = myNoise->GetPerlinFractalSet(
	//	start_x * supergrid_cell_size * brick_size,
	//	start_y * supergrid_cell_size * brick_size, 
	//	0, 
	//	supergrid_cell_size * brick_size, 
	//	supergrid_cell_size * brick_size, 
	//	1);

	auto supercell = std::make_unique<Supercell>();

	supercell->indices.resize(supergrid_cell_size * supergrid_cell_size * supergrid_cell_size, 0);
	for (int z = 0; z < supergrid_cell_size; z++) {
		for (int y = 0; y < supergrid_cell_size; y++) {
			for (int x = 0; x < supergrid_cell_size; x++) {
				Brick brick{};
				bool empty = true;

				uint32_t lod_2x2x2 = 0;
				for (int cell_x = 0; cell_x < brick_size; cell_x++) {
					for (int cell_y = 0; cell_y < brick_size; cell_y++) {
						float height = heights[cell_x + x * brick_size + (cell_y + y * brick_size) * brick_size * supergrid_cell_size];
						//float height = noiseSet[cell_x + x * brick_size + (cell_y + y * brick_size) * brick_size * supergrid_cell_size] * (grid_height / 2.f) + grid_height / 2.f;
						for (int cell_z = 0; cell_z < brick_size; cell_z++) {
							if ((start_z * supergrid_cell_size + z) * brick_size + cell_z < height) {
								uint32_t sub_data = (cell_x + cell_y * brick_size + cell_z * brick_size * brick_size) / (sizeof(uint32_t) * 8);
								uint32_t bit_position = (cell_x + cell_y * brick_size + cell_z * brick_size * brick_size) % (sizeof(uint32_t) * 8);
								brick.data[sub_data] |= (1 << bit_position);
								empty = false;
								lod_2x2x2 |= 1 << (((cell_x & 0b100) >> 2) + ((cell_y & 0b100) >> 1) + (cell_z & 0b100));
							}
						}
					}
				}

				if (!empty) {
					supercell->bricks.push_back(brick);
					// ToDo morton order
					supercell->indices[x + y * supergrid_cell_size + z * supergrid_cell_size * supergrid_cell_size] = (supercell->bricks.size() - 1) | brick_loaded_bit | (lod_2x2x2 << 12);
				}
			}
		}
	}

	auto t = morton(start_x);
	//unsigned int supercell_index = morton(start_x) + (morton(start_y) << 1) + (morton(start_z) << 2);
	//uint32_t supercell_index = MortonEncode3(start_x, start_y, start_z);
	supergrid[start_x + start_y * supergrid_xy + start_z * supergrid_xy * supergrid_xy] = std::move(supercell);
	//supergrid[supercell_index] = std::move(supercell);
	//FastNoiseSIMD::FreeNoiseSet(noiseSet);
}

void Scene::generate() {
	auto begin = std::chrono::steady_clock::now();

	const int thread_count = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(thread_count);

	supergrid.resize(supergrid_z * supergrid_xy * supergrid_xy);

	for (int i = 0; i < thread_count; i++) {
		threads[i] = std::thread([i, thread_count, this]() {
			for (int x = i * (supergrid_xy / thread_count); x < (i + 1) * (supergrid_xy / thread_count); x++) {
				for (int y = 0; y < supergrid_xy; y++) {
					for (int z = 0; z < supergrid_z; z++) {
						generate_supercell(x, y, z);
					}
				}
			}
		});
	}

	for (auto& i : threads) {
		i.join();
	}

	std::cout << "Generation took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();

	// Copy supercell content to GPU
	std::vector<Brick*> gpu_supercells;
	std::vector<uint32_t*> gpu_superindices;
	std::vector<uint32_t> temp_indices(supergrid_cell_size * supergrid_cell_size * supergrid_cell_size);

	for (auto& i : supergrid) {
		for (int j = 0; j < i->indices.size(); j++) {
			if (i->indices[j] & brick_loaded_bit) {
				temp_indices[j] = brick_unloaded_bit | (i->indices[j] & brick_lod_bits);
			} else {
				temp_indices[j] = 0;
			}
		}

		gpu_supercells.push_back(nullptr);
		gpu_superindices.push_back(nullptr);
		// cudaMallocPitch for proper row alignment?
		cuda(Malloc(&gpu_supercells.back(), supergrid_starting_size * sizeof(Brick)));
		cuda(Malloc(&gpu_superindices.back(), i->indices.size() * sizeof(uint32_t)));

		cuda(Memcpy(gpu_superindices.back(), temp_indices.data(), temp_indices.size() * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
		i->gpu_brick_location = gpu_supercells.back();
		i->gpu_indices_location = gpu_superindices.back();
	}

	// Copy pointers for supergrid to GPU
	cuda(Malloc(&gpuScene.bricks, gpu_supercells.size() * sizeof(Brick*)));
	cuda(Memcpy(gpuScene.bricks, gpu_supercells.data(), gpu_supercells.size() * sizeof(Brick*), cudaMemcpyKind::cudaMemcpyHostToDevice));
	cuda(Malloc(&gpuScene.indices, gpu_superindices.size() * sizeof(uint32_t*)));
	cuda(Memcpy(gpuScene.indices, gpu_superindices.data(), gpu_superindices.size() * sizeof(uint32_t*), cudaMemcpyKind::cudaMemcpyHostToDevice));

	cuda(Malloc(&gpuScene.brick_load_queue_count, 4));
	cuda(Memset(gpuScene.brick_load_queue_count, 0, 4));

	cuda(Malloc(&gpuScene.brick_load_queue, brick_load_queue_size * sizeof(glm::vec3)));
	cuda(Memset(gpuScene.brick_load_queue, 0, brick_load_queue_size * sizeof(glm::vec3)));

	cuda(Malloc(&gpuScene.bricks_queue, brick_load_queue_size * sizeof(Brick)));
	cuda(Malloc(&gpuScene.indices_queue, brick_load_queue_size * sizeof(uint32_t)));

	std::cout << "Allocation took " << (std::chrono::steady_clock::now() - begin).count() / 1'000'000 << "ms\n";
	begin = std::chrono::steady_clock::now();
}

int get_supergrid_index(const glm::ivec3& position) {
	return position.x / supergrid_cell_size + position.y / supergrid_cell_size * supergrid_xy + position.z / supergrid_cell_size * supergrid_xy * supergrid_xy;
}

void Scene::process_load_queue() {
	uint32_t brick_to_load_count = 0;
	cuda(Memcpy(&brick_to_load_count, gpuScene.brick_load_queue_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
	brick_to_load_count = std::min(static_cast<uint32_t>(brick_load_queue_size), brick_to_load_count);

	if (brick_to_load_count == 0) {
		return;
	}

	cuda(Memcpy(bricks_to_load, gpuScene.brick_load_queue, brick_to_load_count * sizeof(glm::ivec3), cudaMemcpyDeviceToHost));
	int stage_index = 0;

	//std::cout << "Bricks to load " << brick_to_load_count << "\n";

	// Copy bricks to staging area
	for (int i = 0; i < brick_to_load_count; i++) {
		const glm::ivec3& pos = bricks_to_load[i];

		const auto& supercell = supergrid[get_supergrid_index(pos)];
		const glm::ivec3 block_pos = pos % supergrid_cell_size;
		const uint32_t supercell_index = block_pos.x + block_pos.y * supergrid_cell_size + block_pos.z * supergrid_cell_size * supergrid_cell_size;
		const uint32_t index = supercell->indices[supercell_index];

		brick_gpu_staging[stage_index] = supercell->bricks[index & brick_index_bits];
		indices_gpu_staging[stage_index] = (supercell->gpu_index_highest | brick_loaded_bit | (index & brick_lod_bits));
		supercell->gpu_index_highest++;
		stage_index++;
	}
	cuda(MemcpyAsync(gpuScene.bricks_queue, brick_gpu_staging, brick_to_load_count * sizeof(Brick), cudaMemcpyKind::cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(gpuScene.indices_queue, indices_gpu_staging, brick_to_load_count * sizeof(uint32_t), cudaMemcpyKind::cudaMemcpyHostToDevice));
	
	for (int i = 0; i < brick_to_load_count; i++) {
		const glm::ivec3& pos = bricks_to_load[i];
		const auto& supercell = supergrid[get_supergrid_index(pos)];

		if (supercell->gpu_index_highest >= supercell->gpu_count) {
			Brick* previous = supercell->gpu_brick_location;
			const int new_size = std::pow(2.0, std::ceil(std::log2(supercell->gpu_index_highest + 1)));

			// Grow the storage
			//std::cout << "old size: " << supercell->gpu_count << "\tnew size: " << new_size << "\tneeded: " << supercell->gpu_index_highest << "\n";

			cuda(Malloc(&supercell->gpu_brick_location, new_size * sizeof(Brick)));
			// Copy old content
			cuda(MemcpyAsync(supercell->gpu_brick_location, previous, supercell->gpu_count * sizeof(Brick), cudaMemcpyDeviceToDevice));
			// Update pointer to storage
			cuda(MemcpyAsync(gpuScene.bricks + get_supergrid_index(pos), &supercell->gpu_brick_location, sizeof(Brick*), cudaMemcpyHostToDevice));
			cuda(Free(previous));

			supercell->gpu_count = new_size;
		}
	}
}

void Scene::dump() {
	std::ofstream file("dump.txt");
	for (const auto& i : supergrid) {
		file << i->gpu_index_highest << "\n";
	}
}