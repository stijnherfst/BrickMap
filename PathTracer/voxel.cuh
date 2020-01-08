#pragma once

__device__ __forceinline unsigned int morton(unsigned int x) {
	x = (x ^ (x << 16)) & 0xff0000ff, x = (x ^ (x << 8)) & 0x0300f00f;
	x = (x ^ (x << 4)) & 0x030c30c3, x = (x ^ (x << 2)) & 0x09249249;
	return x;
}

// From http://www.jcgt.org/published/0006/02/01/
__device__ bool intersect_aabb_branchless2(const glm::vec3& origin, const glm::vec3& direction, float& tmin) {
	constexpr glm::vec3 box_min = { 0, 0, 0 };
	constexpr glm::vec3 box_max = { grid_size, grid_size, grid_height };

	const glm::vec3 t1 = (box_min - origin) / direction;
	const glm::vec3 t2 = (box_max - origin) / direction;
	const glm::vec3 tMin = glm::min(t1, t2);
	const glm::vec3 tMax = glm::max(t1, t2);

	tmin = glm::max(glm::max(tMin.x, 0.f), glm::max(tMin.y, tMin.z));
	return glm::min(tMax.x, glm::min(tMax.y, tMax.z)) > tmin;
}

__device__ inline bool intersect_brick(glm::vec3 origin, glm::vec3 direction, glm::vec3& normal, float& distance, Brick* brick) {
	origin = glm::mod(origin, 8.f);
	glm::ivec3 pos = origin;
	glm::ivec3 out;
	glm::ivec3 step;
	glm::vec3 cb, tmax, tdelta;

	cb.x = direction.x > 0 ? pos.x + 1 : pos.x;
	cb.y = direction.y > 0 ? pos.y + 1 : pos.y;
	cb.z = direction.z > 0 ? pos.z + 1 : pos.z;
	out.x = direction.x > 0 ? brick_size : -1;
	out.y = direction.y > 0 ? brick_size : -1;
	out.z = direction.z > 0 ? brick_size : -1;
	step.x = direction.x > 0 ? 1 : -1;
	step.y = direction.y > 0 ? 1 : -1;
	step.z = direction.z > 0 ? 1 : -1;

	float rxr, ryr, rzr;
	if (direction.x != 0) {
		rxr = 1.0f / direction.x;
		tmax.x = (cb.x - origin.x) * rxr;
		tdelta.x = step.x * rxr;
	} else
		tmax.x = 1000000;
	if (direction.y != 0) {
		ryr = 1.0f / direction.y;
		tmax.y = (cb.y - origin.y) * ryr;
		tdelta.y = step.y * ryr;
	} else
		tmax.y = 1000000;
	if (direction.z != 0) {
		rzr = 1.0f / direction.z;
		tmax.z = (cb.z - origin.z) * rzr;
		tdelta.z = step.z * rzr;
	} else
		tmax.z = 1000000;

	distance = 0.f;
	int smallest_component = -1;
	// Stepping through grid
	while (1) {
		int sub_data = (pos.x + pos.y * brick_size + pos.z * brick_size * brick_size) / 32;
		int bit = (pos.x + pos.y * brick_size + pos.z * brick_size * brick_size) % 32;

		if (brick->data[sub_data] & (1 << bit)) {
			if (smallest_component > -1) {
				normal = glm::vec3(0, 0, 0);
				normal[smallest_component] = -step[smallest_component];
			}
			return true;
		}
		smallest_component = (tmax.x < tmax.y) ? ((tmax.x < tmax.z) ? 0 : 2) : ((tmax.y < tmax.z) ? 1 : 2);

		distance = glm::min(tmax.x, glm::min(tmax.y, tmax.z));

		if (tmax.x < tmax.y) {
			if (tmax.x < tmax.z) {
				pos.x += step.x;
				if (pos.x == out.x)
					return false;
				tmax.x += tdelta.x;
			} else {
				pos.z += step.z;
				if (pos.z == out.z)
					return false;
				tmax.z += tdelta.z;
			}
		} else {
			if (tmax.y < tmax.z) {
				pos.y += step.y;
				if (pos.y == out.y)
					return false;
				tmax.y += tdelta.y;
			} else {
				pos.z += step.z;
				if (pos.z == out.z)
					return false;
				tmax.z += tdelta.z;
			}
		}
	}
	return false;
}
__device__ inline bool intersect_brick_simple(glm::vec3 origin, glm::vec3 direction, Brick* brick) {
	origin = glm::mod(origin, 8.f);
	glm::ivec3 pos = origin;
	glm::ivec3 out;
	glm::ivec3 step;
	glm::vec3 cb, tmax, tdelta;

	cb.x = direction.x > 0 ? pos.x + 1 : pos.x;
	cb.y = direction.y > 0 ? pos.y + 1 : pos.y;
	cb.z = direction.z > 0 ? pos.z + 1 : pos.z;
	out.x = direction.x > 0 ? brick_size : -1;
	out.y = direction.y > 0 ? brick_size : -1;
	out.z = direction.z > 0 ? brick_size : -1;
	step.x = direction.x > 0 ? 1 : -1;
	step.y = direction.y > 0 ? 1 : -1;
	step.z = direction.z > 0 ? 1 : -1;

	float rxr, ryr, rzr;
	if (direction.x != 0) {
		rxr = 1.0f / direction.x;
		tmax.x = (cb.x - origin.x) * rxr;
		tdelta.x = step.x * rxr;
	} else
		tmax.x = 1000000;
	if (direction.y != 0) {
		ryr = 1.0f / direction.y;
		tmax.y = (cb.y - origin.y) * ryr;
		tdelta.y = step.y * ryr;
	} else
		tmax.y = 1000000;
	if (direction.z != 0) {
		rzr = 1.0f / direction.z;
		tmax.z = (cb.z - origin.z) * rzr;
		tdelta.z = step.z * rzr;
	} else
		tmax.z = 1000000;

	// Stepping through grid
	while (1) {
		int sub_data = (pos.x + pos.y * brick_size + pos.z * brick_size * brick_size) / 32;
		int bit = (pos.x + pos.y * brick_size + pos.z * brick_size * brick_size) % 32;
		if (brick->data[sub_data] & (1 << bit)) {
			return true;
		}

		if (tmax.x < tmax.y) {
			if (tmax.x < tmax.z) {
				pos.x += step.x;
				if (pos.x == out.x)
					return false;
				tmax.x += tdelta.x;
			} else {
				pos.z += step.z;
				if (pos.z == out.z)
					return false;
				tmax.z += tdelta.z;
			}
		} else {
			if (tmax.y < tmax.z) {
				pos.y += step.y;
				if (pos.y == out.y)
					return false;
				tmax.y += tdelta.y;
			} else {
				pos.z += step.z;
				if (pos.z == out.z)
					return false;
				tmax.z += tdelta.z;
			}
		}
	}
	return false;
}

__device__ inline bool intersect_voxel(glm::vec3 origin, glm::vec3 direction, glm::vec3& normal, float& distance, Scene::GPUScene scene) {
	float tminn;
	if (!intersect_aabb_branchless2(origin, direction, tminn)) {
		return false;
	}

	// Move ray to hitpoint
	if (tminn > 0) {
		origin += direction * tminn;

		constexpr glm::vec3 scale = (glm::vec3(1.f / (grid_size / static_cast<float>(grid_height)), 1.f / (grid_size / static_cast<float>(grid_height)), 1.f / (grid_height / static_cast<float>(grid_height))));
		constexpr glm::vec3 grid_center = glm::vec3(grid_size / 2.f, grid_size / 2.f, grid_height / 2.f);

		glm::vec3 to_center = glm::abs(grid_center - origin) * scale;
		glm::vec3 signs = glm::sign(origin - grid_center);
		to_center /= glm::max(to_center.x, glm::max(to_center.y, to_center.z));
		normal = signs * glm::trunc(to_center + 0.000001f);

		origin += -normal * epsilon;
	}
	origin /= 8.f;
	glm::ivec3 pos = origin;
	glm::ivec3 out;
	glm::ivec3 step;
	glm::vec3 cb, tmax, tdelta;

	// Needed because sometimes the AABB intersect returns true while the ray is actually outside slightly. Only happens for faces that touch the AABB sides
	if (pos.x < 0 || pos.x >= cells || pos.y < 0 || pos.y >= cells || pos.z < 0 || pos.z >= cells_height) {
		return false;
	}

	cb.x = direction.x > 0 ? pos.x + 1 : pos.x;
	cb.y = direction.y > 0 ? pos.y + 1 : pos.y;
	cb.z = direction.z > 0 ? pos.z + 1 : pos.z;
	out.x = direction.x > 0 ? cells : -1;
	out.y = direction.y > 0 ? cells : -1;
	out.z = direction.z > 0 ? cells_height : -1;
	step.x = direction.x > 0 ? 1 : -1;
	step.y = direction.y > 0 ? 1 : -1;
	step.z = direction.z > 0 ? 1 : -1;

	float rxr, ryr, rzr;
	if (direction.x != 0) {
		rxr = 1.0f / direction.x;
		tmax.x = (cb.x - origin.x) * rxr;
		tdelta.x = step.x * rxr;
	} else
		tmax.x = 1000000;
	if (direction.y != 0) {
		ryr = 1.0f / direction.y;
		tmax.y = (cb.y - origin.y) * ryr;
		tdelta.y = step.y * ryr;
	} else
		tmax.y = 1000000;
	if (direction.z != 0) {
		rzr = 1.0f / direction.z;
		tmax.z = (cb.z - origin.z) * rzr;
		tdelta.z = step.z * rzr;
	} else
		tmax.z = 1000000;

	float new_distance = 0.f;

	int smallest_component = -1;
	// Stepping through grid
	while (1) {

		//uint32_t index = scene.indices[supercell_index][X % supergrid_cell_size + (Y % supergrid_cell_size) * supergrid_cell_size + (Z % supergrid_cell_size) * supergrid_cell_size * supergrid_cell_size];
		//if (index & brick_loaded_bit) {
		//	Brick* p = scene.bricks[supercell_index];
		//	if (intersect_brick_simple(origin * 8.f + direction * (distance * 8.f + epsilon), direction, &p[index & 0x7FFFFFFFu])) {
		//		return true;
		//	}
		//} else if (index & brick_unloaded_bit) {
		//	// load?
		//	return true;
		//}


		int supercell_index = pos.x / supergrid_cell_size + (pos.y / supergrid_cell_size) * supergrid_xy + (pos.z / supergrid_cell_size) * supergrid_xy * supergrid_xy;
		//uint32_t& index = scene.brick_grid[morton(pos.x) + (morton(pos.y) << 1) + (morton(pos.z) << 2)];
		uint32_t& index = scene.indices[supercell_index][(pos.x % supergrid_cell_size) + (pos.y % supergrid_cell_size) * supergrid_cell_size + (pos.z % supergrid_cell_size) * supergrid_cell_size * supergrid_cell_size];
		if (index & brick_loaded_bit) {
			distance = new_distance * 8.f + tminn + epsilon;
			//return true;

			float sub_distance = 0.f;

			if (smallest_component > -1) {
				normal = glm::vec3(0, 0, 0);
				normal[smallest_component] = -step[smallest_component];
			}

			Brick* p = scene.bricks[supercell_index];


			if (intersect_brick(origin * 8.f + direction * (new_distance * 8.f + epsilon), direction, normal, sub_distance, &p[index & brick_data_bits])) {
				distance = new_distance * 8.f + sub_distance + tminn + epsilon;
				return true;
			}
		} else if (index & brick_requested_bit) {
			return false;
		} else if (index & brick_unloaded_bit) {
			uint32_t old = atomicOr(&index, brick_requested_bit);

			if (!(old & brick_requested_bit)) {
				// request chunk to be loaded
				
				const unsigned int load_index = atomicAdd(scene.brick_load_queue_count, 1);
				if (load_index < brick_load_queue_size) {
					scene.brick_load_queue[load_index] = pos;
				} else {
					atomicAnd(&index, ~brick_requested_bit);
					//printf("haaaaaaaaaaaaaa\n");
					// happens a lot. Fix?
				}
			}
			//distance = 256;
			//return false;
			if (smallest_component > -1) {
				normal = glm::vec3(0, 0, 0);
				normal[smallest_component] = -step[smallest_component];
			}
			return true;
		} 

		smallest_component = (tmax.x < tmax.y) ? ((tmax.x < tmax.z) ? 0 : 2) : ((tmax.y < tmax.z) ? 1 : 2);

		if (tmax.x < tmax.y) {
			if (tmax.x < tmax.z) {
				pos.x += step.x;
				if (pos.x == out.x)
					return false;
				new_distance = tmax.x;
				tmax.x += tdelta.x;
			} else {
				pos.z += step.z;
				if (pos.z == out.z)
					return false;
				new_distance = tmax.z;
				tmax.z += tdelta.z;
			}
		} else {
			if (tmax.y < tmax.z) {
				pos.y += step.y;
				if (pos.y == out.y)
					return false;
				new_distance = tmax.y;
				tmax.y += tdelta.y;
			} else {
				pos.z += step.z;
				if (pos.z == out.z)
					return false;
				new_distance = tmax.z;
				tmax.z += tdelta.z;
			}
		}
	}
	return false;
}

__device__ inline bool intersect_voxel_simple(glm::vec3 origin, glm::vec3 direction, Scene::GPUScene scene) {
	// Check if ray hits grid AABB
	float tminn;
	if (!intersect_aabb_branchless2(origin, direction, tminn)) {
		return false;
	}

	// Move ray to hitpoint
	if (tminn > 0) {
		origin += direction * tminn;

		constexpr glm::vec3 scale = (glm::vec3(1.f / (grid_size / static_cast<float>(grid_height)), 1.f / (grid_size / static_cast<float>(grid_height)), 1.f / (grid_height / static_cast<float>(grid_height))));
		constexpr glm::vec3 grid_center = glm::vec3(grid_size / 2.f, grid_size / 2.f, grid_height / 2.f);

		glm::vec3 to_center = glm::abs(grid_center - origin) * scale;
		glm::vec3 signs = glm::sign(origin - grid_center);
		to_center /= glm::max(to_center.x, glm::max(to_center.y, to_center.z));

		origin += -(signs * glm::trunc(to_center + 0.000001f)) * epsilon;
	}
	origin /= 8.f;

	// Initialize
	glm::vec3 cb, tmax, tdelta;
	int stepX, outX, X = ((int)origin.x);
	int stepY, outY, Y = ((int)origin.y);
	int stepZ, outZ, Z = ((int)origin.z);

	// Needed because sometimes the AABB intersect returns true while the ray is actually outside slightly. Only happens for faces that touch the AABB sides
	if (X < 0 || X >= cells || Y < 0 || Y >= cells || Z < 0 || Z >= cells_height) {
		return false;
	}

	if (direction.x > 0) {
		stepX = 1;
		outX = cells;
		cb.x = (X + 1);
	} else {
		stepX = -1;
		outX = -1;
		cb.x = X;
	}
	if (direction.y > 0.0f) {
		stepY = 1;
		outY = cells;
		cb.y = (Y + 1);
	} else {
		stepY = -1;
		outY = -1;
		cb.y = Y;
	}
	if (direction.z > 0.0f) {
		stepZ = 1;
		outZ = cells_height;
		cb.z = (Z + 1);
	} else {
		stepZ = -1;
		outZ = -1;
		cb.z = Z;
	}
	float rxr, ryr, rzr;
	if (direction.x != 0) {
		rxr = 1.0f / direction.x;
		tmax.x = (cb.x - origin.x) * rxr;
		tdelta.x = stepX * rxr;
	} else
		tmax.x = 1000000;
	if (direction.y != 0) {
		ryr = 1.0f / direction.y;
		tmax.y = (cb.y - origin.y) * ryr;
		tdelta.y = stepY * ryr;
	} else
		tmax.y = 1000000;
	if (direction.z != 0) {
		rzr = 1.0f / direction.z;
		tmax.z = (cb.z - origin.z) * rzr;
		tdelta.z = stepZ * rzr;
	} else
		tmax.z = 1000000;
	float distance = 0.f;

	// Stepping through grid
	while (1) {
		int supercell_index = X / supergrid_cell_size + (Y / supergrid_cell_size) * supergrid_xy + (Z / supergrid_cell_size) * supergrid_xy * supergrid_xy;
		//uint32_t& index = scene.brick_grid[morton(pos.x) + (morton(pos.y) << 1) + (morton(pos.z) << 2)];
		uint32_t& index = scene.indices[supercell_index][X % supergrid_cell_size + (Y % supergrid_cell_size) * supergrid_cell_size + (Z % supergrid_cell_size) * supergrid_cell_size * supergrid_cell_size];
		if (index & brick_loaded_bit) {
			float sub_distance = 0.f;

			Brick* p = scene.bricks[supercell_index];

			//distance = new_distance * 8.f + tminn + epsilon;
			//return true;
			if (index & brick_unloaded_bit) {
				printf("uhoh\n");
			}
			if (intersect_brick_simple(origin * 8.f + direction * (distance * 8.f + epsilon), direction, &p[index & brick_loaded_rest])) {
				return true;
			}
		} else if (index & brick_unloaded_bit) {
			//return true;
			//auto t = &index;
			//bool aligned = t & 0b11u;

			//uint32_t old = atomicOr(reinterpret_cast<uint32_t*>(&index - aligned), brick_requested_bit << (aligned * 2));
			uint32_t old = atomicOr(&index, brick_requested_bit);

			if (!(old & brick_requested_bit)) {
				// request chunk to be loaded
				const unsigned int load_index = atomicAdd(scene.brick_load_queue_count, 1);
				if (load_index < brick_load_queue_size) {
					scene.brick_load_queue[load_index] = glm::ivec3(X, Y, Z);
				}
			}

			return true;
		}


		//uint32_t index = scene.brick_grid[morton(X) + (morton(Y) << 1) + (morton(Z) << 2)];
		//int supercell_index = X / supergrid_cell_size + (Y / supergrid_cell_size) * supergrid_xy + (Z / supergrid_cell_size) * supergrid_xy * supergrid_xy;
		//uint32_t index = scene.indices[supercell_index][X % supergrid_cell_size + (Y % supergrid_cell_size) * supergrid_cell_size + (Z % supergrid_cell_size) * supergrid_cell_size * supergrid_cell_size];
		//if (index & brick_loaded_bit) {
		//	//return true;

		//	Brick* p = scene.bricks[supercell_index];
		//	if (intersect_brick_simple(origin * 8.f + direction * (distance * 8.f + epsilon), direction, &p[index & 0x7FFFu])) {
		//		return true;
		//	}
		//}
		//else if (index & brick_unloaded_bit) {
			// load?
		//	return true;
		//}

		if (tmax.x < tmax.y) {
			if (tmax.x < tmax.z) {
				X += stepX;
				if (X == outX)
					return false;
				distance = tmax.x;
				tmax.x += tdelta.x;
			} else {
				Z += stepZ;
				if (Z == outZ)
					return false;
				distance = tmax.z;
				tmax.z += tdelta.z;
			}
		} else {
			if (tmax.y < tmax.z) {
				Y += stepY;
				if (Y == outY)
					return false;
				distance = tmax.y;
				tmax.y += tdelta.y;
			} else {
				Z += stepZ;
				if (Z == outZ)
					return false;
				distance = tmax.z;
				tmax.z += tdelta.z;
			}
		}
	}
	return false;
}