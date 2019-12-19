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
	};

	GPUScene gpuScene;

	struct Supercell {
		std::vector<Brick> bricks;
		std::vector<uint32_t> indices;
	};

	std::vector<Supercell> supergrid;

	void generate_supercell(int start_x, int start_y, int start_z);
	void generate();
	void process_load_queue();
};
