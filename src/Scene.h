#pragma once

struct Brick {
	uint32_t data[cell_members];
};

class Scene {
public:
	struct GPUScene {
		uint32_t** indices;
		Brick** bricks;
		glm::ivec3* brick_load_queue;
		uint32_t* brick_load_queue_count;

		Brick* bricks_queue;
		uint32_t* indices_queue;
	};

	GPUScene gpuScene;

	struct Supercell {
		int gpu_count = supergrid_starting_size;
		std::vector<Brick> bricks;
		std::vector<uint32_t> indices;
		std::vector<uint32_t> gpu_indices;
		uint32_t* gpu_indices_location;
		Brick* gpu_brick_location;
		int gpu_index_highest = 0;
	};

	std::vector<std::unique_ptr<Supercell>> supergrid;

	glm::ivec3* bricks_to_load;
	Brick* brick_gpu_staging;
	uint32_t* indices_gpu_staging;

	Scene();
	~Scene();

	void generate_supercell(int start_x, int start_y, int start_z);
	void generate();
	void process_load_queue();
	void dump();
};
