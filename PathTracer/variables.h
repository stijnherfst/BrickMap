#pragma once

constexpr static float pi = 3.1415926535897932f;
constexpr static float inv_pi = 1.0f / pi;

constexpr static unsigned window_width = 1920;
constexpr static unsigned window_height = 1080;

constexpr static unsigned render_width = 1920;
constexpr static unsigned render_height = 1080;

constexpr static int grid_size = 16384;
constexpr static int grid_height = 256;
constexpr static int brick_size = 8;

constexpr static int supergrid_cell_size = 16;
constexpr static int supergrid_xy = grid_size / brick_size / supergrid_cell_size;
constexpr static int supergrid_z = grid_height / brick_size / supergrid_cell_size;


constexpr static int cells = grid_size / brick_size;
constexpr static int cells_height = grid_height / brick_size;
// The amount of uint32_t members holding voxel bit data
constexpr static int cell_members = brick_size * brick_size * brick_size / 32;

constexpr static float epsilon = 0.001f;

//constexpr static uint32_t brick_loaded_bit = 0x80000000u;
constexpr static uint32_t brick_loaded_bit = 0x8000u;
constexpr static uint32_t brick_unloaded_bit = 0x40000000u;
constexpr static uint32_t brick_requested_bit = 0x20000000u;

constexpr static int brick_load_queue_size = 4096;

extern glm::vec2 sun_position;
extern bool sun_position_changed;
extern int sm_cores;

struct RayQueue {
	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec3 direct;
	glm::vec3 normal;
	float distance;
	int identifier;
	int bounces;
	unsigned int pixel_index;
};

struct ShadowQueue {
	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec3 color;
	int pixel_index;
};

const unsigned int ray_queue_buffer_size = 2 * 1'048'576;
