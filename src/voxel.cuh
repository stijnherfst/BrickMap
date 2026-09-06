#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/gtx/vec_swizzle.hpp"

__device__ __forceinline unsigned int morton(unsigned int x) {
	x = (x ^ (x << 16)) & 0xff0000ff, x = (x ^ (x << 8)) & 0x0300f00f;
	x = (x ^ (x << 4)) & 0x030c30c3, x = (x ^ (x << 2)) & 0x09249249;
	return x;
}

// Use read-only cache __ldg instead of L1
template<typename T>
__device__ __forceinline T* dda_load_ptr(T* const* p) {
	return reinterpret_cast<T*>(__ldg(reinterpret_cast<const unsigned long long*>(p)));
}

// Ray-invariant DDA constants
struct DdaRay {
	glm::vec3 tdelta; // |1/d|, which is sign(d)/d for every component including a zero one
	glm::ivec3 step;
	bool nonzero_x, nonzero_y, nonzero_z;
};

__device__ __forceinline DdaRay make_dda_ray(const glm::vec3& direction) {
	DdaRay ray;
	ray.tdelta.x = __fdividef(1.f, direction.x == 0.f ? 1e-20f : fabsf(direction.x));
	ray.tdelta.y = __fdividef(1.f, direction.y == 0.f ? 1e-20f : fabsf(direction.y));
	ray.tdelta.z = __fdividef(1.f, direction.z == 0.f ? 1e-20f : fabsf(direction.z));
	ray.step.x = direction.x > 0.f ? 1 : -1;
	ray.step.y = direction.y > 0.f ? 1 : -1;
	ray.step.z = direction.z > 0.f ? 1 : -1;
	ray.nonzero_x = direction.x != 0.f;
	ray.nonzero_y = direction.y != 0.f;
	ray.nonzero_z = direction.z != 0.f;
	return ray;
}

// From http://www.jcgt.org/published/0006/02/01/
__device__ inline bool intersect_aabb_branchless2(const glm::vec3& origin, const glm::vec3& inv, float& tmin, int& entry_axis) {
	constexpr glm::vec3 box_max = {grid_size, grid_size, grid_height};

	const glm::vec3 t1 = -origin * inv;
	const glm::vec3 t2 = (box_max - origin) * inv;
	const glm::vec3 tMin = glm::min(t1, t2);
	const glm::vec3 tMax = glm::max(t1, t2);

	tmin = glm::max(glm::max(tMin.x, 0.f), glm::max(tMin.y, tMin.z));
	entry_axis = (tMin.x > tMin.y) ? ((tMin.x > tMin.z) ? 0 : 2) : ((tMin.y > tMin.z) ? 1 : 2);
	return glm::min(tMax.x, glm::min(tMax.y, tMax.z)) > tmin;
}

// Which axis steps next, and the t at which the cell being left was entered.
struct StepPick {
	bool mx, my, mz;
	float t_entry;
};

__device__ __forceinline StepPick pick_axis(const glm::vec3& tmax) {
	const bool xy = tmax.x < tmax.y;
	const float t_xy = xy ? tmax.x : tmax.y;
	const bool q = t_xy < tmax.z;
	return {xy && q, !xy && q, !q, q ? t_xy : tmax.z};
}

// The face the ray entered the current cell through, i.e. the opposite of the step just
// taken
__device__ __forceinline glm::vec3 face_normal(const StepPick& s, const glm::vec3& direction) {
	glm::vec3 n;
	n.x = s.mx ? (direction.x > 0.f ? -1.f : 1.f) : 0.f;
	n.y = s.my ? (direction.y > 0.f ? -1.f : 1.f) : 0.f;
	n.z = s.mz ? (direction.z > 0.f ? -1.f : 1.f) : 0.f;
	return n;
}

// tmax for one axis
__device__ __forceinline float axis_tmax(bool nonzero, int step, float frac, float tdelta) {
	return nonzero ? (step > 0 ? 1.f - frac : frac) * tdelta : 1000000.f;
}

// Occupancy predicates for the two sub grids
struct ByteField {
	uint32_t bits;

	__device__ __forceinline bool operator()(const glm::ivec3& p) const {
		return bits & (1u << (p.x + p.y * 2 + p.z * 4));
	}
};

struct BrickField {
	const Brick* brick;

	__device__ __forceinline bool operator()(const glm::ivec3& p) const {
		const int voxel = p.x + p.y * brick_size + p.z * brick_size * brick_size;
		return __ldg(&brick->data[voxel / 32]) & (1u << (voxel % 32));
	}
};

// One DDA over an N x N x N sub-grid, N a power of two.
template<int N, bool AnyHit, typename Field>
__device__ inline bool
dda_sub(glm::vec3 origin, const glm::vec3& direction, const DdaRay& ray, const Field& field, glm::vec3& normal, float& distance) {
	glm::ivec3 pos = origin; // should be glm::floor()?
	const glm::vec3 frac = origin - glm::vec3(pos);

	glm::vec3 tmax;
	tmax.x = axis_tmax(ray.nonzero_x, ray.step.x, frac.x, ray.tdelta.x);
	tmax.y = axis_tmax(ray.nonzero_y, ray.step.y, frac.y, ray.tdelta.y);
	tmax.z = axis_tmax(ray.nonzero_z, ray.step.z, frac.z, ray.tdelta.z);

	pos = glm::clamp(pos % N, 0, N - 1);

	StepPick s {false, false, false, 0.f};
	while (true) {
		if (field(pos)) {
			if constexpr (!AnyHit) {
				if (s.mx || s.my || s.mz) {
					normal = face_normal(s, direction);
				}
				distance = s.t_entry;
			}
			return true;
		}

		s = pick_axis(tmax);

		pos.x += s.mx ? ray.step.x : 0;
		pos.y += s.my ? ray.step.y : 0;
		pos.z += s.mz ? ray.step.z : 0;
		if ((pos.x | pos.y | pos.z) & ~(N - 1)) {
			break;
		}
		tmax.x += s.mx ? ray.tdelta.x : 0.f;
		tmax.y += s.my ? ray.tdelta.y : 0.f;
		tmax.z += s.mz ? ray.tdelta.z : 0.f;
	}
	if constexpr (!AnyHit) {
		distance = 0.f;
	}
	return false;
}

// Top-level traversal over the grid of 8x8x8 voxel cells
template<bool AnyHit>
__device__ inline bool dda_cells(
	glm::vec3 origin,
	const glm::vec3 direction,
	Scene::GPUScene scene,
	glm::ivec3 camera_position,
	glm::vec3& normal,
	float& distance
) {
	const DdaRay ray = make_dda_ray(direction);

	float tminn = 0.f;
	if (origin.x <= 0.f || origin.y <= 0.f || origin.z <= 0.f || origin.x >= grid_size || origin.y >= grid_size
		|| origin.z >= grid_height) {
		int entry_axis;

		const glm::vec3 ray_inv = {
			direction.x < 0.f ? -ray.tdelta.x : ray.tdelta.x,
			direction.y < 0.f ? -ray.tdelta.y : ray.tdelta.y,
			direction.z < 0.f ? -ray.tdelta.z : ray.tdelta.z
		};

		if (!intersect_aabb_branchless2(origin, ray_inv, tminn, entry_axis)) {
			return false;
		}

		// Move ray to hitpoint
		if (tminn > 0.f) {
			if constexpr (AnyHit) {
				origin += direction * (tminn + epsilon);
			} else {
				origin += direction * tminn;
				normal = glm::vec3(0.f);
				normal[entry_axis] = direction[entry_axis] > 0.f ? -1.f : 1.f;
				origin -= normal * epsilon;
			}
		}

		// Needed because sometimes the AABB intersect returns true while the ray is actually
		// outside slightly. Only happens for faces that touch the AABB sides.
		if (origin.x < 0.f || origin.y < 0.f || origin.z < 0.f || origin.x >= grid_size || origin.y >= grid_size
			|| origin.z >= grid_height) {
			return false;
		}
	}

	origin /= 8.f;
	glm::ivec3 pos = glm::ivec3(origin);

	const glm::vec3 frac = origin - glm::vec3(pos);
	glm::vec3 tmax;
	tmax.x = axis_tmax(ray.nonzero_x, ray.step.x, frac.x, ray.tdelta.x);
	tmax.y = axis_tmax(ray.nonzero_y, ray.step.y, frac.y, ray.tdelta.y);
	tmax.z = axis_tmax(ray.nonzero_z, ray.step.z, frac.z, ray.tdelta.z);

	StepPick s {false, false, false, 0.f};

	int cached_supercell = -1;
	uint32_t* cell_indices = nullptr;

	while (true) {
		const unsigned int ux = pos.x;
		const unsigned int uy = pos.y;
		const unsigned int uz = pos.z;

		const unsigned int lx = ux % supergrid_cell_size;
		const unsigned int ly = uy % supergrid_cell_size;
		const unsigned int lz = uz % supergrid_cell_size;

		const int supercell_index =
			ux / supergrid_cell_size + (uy / supergrid_cell_size) * supergrid_xy + (uz / supergrid_cell_size) * supergrid_xy * supergrid_xy;
		if (supercell_index != cached_supercell) {
			cached_supercell = supercell_index;
			cell_indices = dda_load_ptr(&scene.indices[supercell_index]);

			// Empty supercell space skip
			if (__ldg(&scene.supercell_empty[supercell_index])) {
				const int nx = ray.step.x > 0 ? static_cast<int>(supergrid_cell_size - lx) : static_cast<int>(lx) + 1;
				const int ny = ray.step.y > 0 ? static_cast<int>(supergrid_cell_size - ly) : static_cast<int>(ly) + 1;
				const int nz = ray.step.z > 0 ? static_cast<int>(supergrid_cell_size - lz) : static_cast<int>(lz) + 1;
				const float ex = tmax.x + (nx - 1) * ray.tdelta.x;
				const float ey = tmax.y + (ny - 1) * ray.tdelta.y;
				const float ez = tmax.z + (nz - 1) * ray.tdelta.z;
				const float t_exit = fminf(ex, fminf(ey, ez));

				const int kx = t_exit > tmax.x ? min(static_cast<int>((t_exit - tmax.x) * fabsf(direction.x)) + 1, nx - 1) : 0;
				const int ky = t_exit > tmax.y ? min(static_cast<int>((t_exit - tmax.y) * fabsf(direction.y)) + 1, ny - 1) : 0;
				const int kz = t_exit > tmax.z ? min(static_cast<int>((t_exit - tmax.z) * fabsf(direction.z)) + 1, nz - 1) : 0;

				pos.x += kx * ray.step.x;
				pos.y += ky * ray.step.y;
				pos.z += kz * ray.step.z;
				tmax.x += kx * ray.tdelta.x;
				tmax.y += ky * ray.tdelta.y;
				tmax.z += kz * ray.tdelta.z;
				continue;
			}
		}
		const unsigned int cell_offset = lx + ly * supergrid_cell_size + lz * supergrid_cell_size * supergrid_cell_size;
		const uint32_t index = __ldg(&cell_indices[cell_offset]);

		if (index) {
			if constexpr (!AnyHit) {
				if (s.mx || s.my || s.mz) {
					normal = face_normal(s, direction);
				} else if (tminn <= 0.f) {
					// The ray started inside a non-empty cell, Face the ray instead so shading stays defined.
					normal = -direction;
				}
			}

			const glm::ivec3 difference = camera_position - pos;
			const int lod_distance_squared = difference.x * difference.x + difference.y * difference.y + difference.z * difference.z;
			float sub_distance = 0.f;

			if (lod_distance_squared > lod_distance_8x8x8) {
				if constexpr (!AnyHit) {
					distance = s.t_entry * 8.f + tminn;
				}
				return true;
			} else if (lod_distance_squared > lod_distance_2x2x2) {
				// For some reason the normal displacement has to be made even smaller
				glm::vec3 sub_origin = (origin + direction * s.t_entry) * 2.f;
				if constexpr (AnyHit) {
					sub_origin += direction * 0.2f * epsilon;
				} else {
					sub_origin -= normal * 0.2f * epsilon;
				}
				if (dda_sub<2, AnyHit, ByteField>(
						sub_origin,
						direction,
						ray,
						ByteField {(index & brick_lod_bits) >> 12},
						normal,
						sub_distance
					)) {
					if constexpr (!AnyHit) {
						distance = s.t_entry * 8.f + sub_distance * 4.f + tminn;
					}
					return true;
				}
			} else {
				if (index & brick_loaded_bit) {
					glm::vec3 sub_origin = (origin + direction * s.t_entry) * 8.f;
					if constexpr (AnyHit) {
						sub_origin += direction * epsilon;
					} else {
						sub_origin -= normal * epsilon;
					}

					// First DDA the LOD for a quick discard
					if (dda_sub<2, true, ByteField>(
							sub_origin * 0.25f,
							direction,
							ray,
							ByteField {(index & brick_lod_bits) >> 12},
							normal,
							sub_distance
						)) {
						const Brick* p = dda_load_ptr(&scene.bricks[cached_supercell]);
						if (dda_sub<brick_size, AnyHit, BrickField>(
								sub_origin,
								direction,
								ray,
								BrickField {&p[index & brick_index_bits]},
								normal,
								sub_distance
							)) {
							if constexpr (!AnyHit) {
								distance = s.t_entry * 8.f + sub_distance + tminn;
							}
							return true;
						}
					}
				} else if (index & brick_unloaded_bit) {
					uint32_t old = atomicOr(&cell_indices[cell_offset], brick_requested_bit);

					if (!(old & brick_requested_bit)) {
						// request chunk to be loaded
						const unsigned int load_index = atomicAdd(scene.brick_load_queue_count, 1);
						if (load_index < brick_load_queue_size) {
							scene.brick_load_queue[load_index] = pos;
						} else {
							atomicAnd(&cell_indices[cell_offset], ~brick_requested_bit);
							// ToDo happens a lot. Fix?
						}
					}

					if constexpr (!AnyHit) {
						distance = s.t_entry * 8.f + tminn;
					}
					return true;
				}
			}
		}

		s = pick_axis(tmax);

		pos.x += s.mx ? ray.step.x : 0;
		pos.y += s.my ? ray.step.y : 0;
		pos.z += s.mz ? ray.step.z : 0;
		if (((pos.x | pos.y) & ~(cells - 1)) | (pos.z & ~(cells_height - 1))) {
			break;
		}
		tmax.x += s.mx ? ray.tdelta.x : 0.f;
		tmax.y += s.my ? ray.tdelta.y : 0.f;
		tmax.z += s.mz ? ray.tdelta.z : 0.f;
	}
	return false;
}

__device__ inline bool intersect_voxel(
	glm::vec3 origin,
	const glm::vec3 direction,
	glm::vec3& normal,
	float& distance,
	Scene::GPUScene scene,
	glm::ivec3 camera_position
) {
	return dda_cells<false>(origin, direction, scene, camera_position, normal, distance);
}

__device__ inline bool intersect_voxel_any(glm::vec3 origin, const glm::vec3 direction, Scene::GPUScene scene, glm::ivec3 camera_position) {
	glm::vec3 normal;
	float distance;
	return dda_cells<true>(origin, direction, scene, camera_position, normal, distance);
}
