#include "stdafx.h"
#include "sunsky.cuh"

#include "assert_cuda.h"
#include "cuda_surface_types.h"
#include "device_launch_parameters.h"
#include "surface_functions.h"

#include "cuda_definitions.h"

constexpr int NUM_SPHERES = 7;
constexpr float VERY_FAR = 1e20f;
constexpr int MAX_BOUNCES = 5;

surface<void, cudaSurfaceType2D> surf;
texture<float, cudaTextureTypeCubemap> skybox;

//"Xorshift RNGs" by George Marsaglia
//http://excamera.com/sphinx/article-xorshift.html
__device__ unsigned int RandomInt(unsigned int& seed) {
	seed ^= seed << 13;
	seed ^= seed >> 17;
	seed ^= seed << 5;
	return seed;
}

//Random float between [0,1).
__device__ float RandomFloat(unsigned int& seed) {
	return RandomInt(seed) * 2.3283064365387e-10f;
}

__device__ float RandomFloat2(unsigned int& seed) {
	return (RandomInt(seed) >> 16) / 65535.0f;
}

__device__ int RandomIntBetween0AndMax(unsigned int& seed, int max) {
	return int(RandomFloat(seed) * (max + 0.99999f));
}

// A 100% correct but slow implementation
__device__ bool intersect_aabb_correct(const RayQueue& ray, float& tmin) {
	glm::vec3 box_min = { 0, 0, 0 };
	glm::vec3 box_max = { grid_size, grid_size, grid_size };

	tmin = (box_min[0] - ray.origin[0]) / ray.direction.x;
	float tmax = (box_max[0] - ray.origin[0]) / ray.direction.x;

	if (tmin > tmax) {
		float a = tmin;
		tmin = tmax;
		tmax = a;
	}

	float tymin = (box_min[1] - ray.origin[1]) / ray.direction.y;
	float tymax = (box_max[1] - ray.origin[1]) / ray.direction.y;

	if (tymin > tymax) {
		float a = tymin;
		tymin = tymax;
		tymax = a;
	}

	if ((tmin > tymax) || (tymin > tmax))
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = (box_min[2] - ray.origin[2]) / ray.direction.z;
	float tzmax = (box_max[2] - ray.origin[2]) / ray.direction.z;

	if (tzmin > tzmax) {
		float a = tzmin;
		tzmin = tzmax;
		tzmax = a;
	}

	if ((tmin > tzmax) || (tzmin > tmax))
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return tmax > glm::max(tmin, 0.f);
}

__device__ inline bool intersect_aabb_branchless(const RayQueue& ray, float& tmin) {
	glm::vec3 box_min = { 0, 0, 0 };
	glm::vec3 box_max = { grid_size, grid_size, grid_size };
	glm::vec3 dir_inv = 1.f / ray.direction;

	float t1 = (box_min[0] - ray.origin[0]) * dir_inv[0];
	float t2 = (box_max[0] - ray.origin[0]) * dir_inv[0];

	tmin = glm::min(t1, t2);
	float tmax = glm::max(t1, t2);

	for (int i = 1; i < 3; ++i) {
		t1 = (box_min[i] - ray.origin[i]) * dir_inv[i];
		t2 = (box_max[i] - ray.origin[i]) * dir_inv[i];

		//tmin = glm::max(tmin, glm::min(t1, t2));
		//tmax = glm::min(tmax, glm::max(t1, t2));
		tmin = glm::max(tmin, glm::min(glm::min(t1, t2), tmax));
		tmax = glm::min(tmax, glm::max(glm::max(t1, t2), tmin));
	}

	return tmax > glm::max(tmin, 0.f);
}

//// From http://www.jcgt.org/published/0006/02/01/
//__device__ bool intersect_aabb_branchless2(const RayQueue& ray, float& tmin) {
//	constexpr glm::vec3 box_min = { 0, 0, 0 };
//	constexpr glm::vec3 box_max = { grid_size, grid_size, grid_size };
//
//	const glm::vec3 t1 = (box_min - ray.origin) / ray.direction;
//	const glm::vec3 t2 = (box_max - ray.origin) / ray.direction;
//	const glm::vec3 tMin = glm::min(t1, t2);
//	const glm::vec3 tMax = glm::max(t1, t2);
//
//	tmin = glm::max(glm::max(tMin.x, 0.f), glm::max(tMin.y, tMin.z));
//	return glm::min(tMax.x, glm::min(tMax.y, tMax.z)) > tmin;
//}

// From http://www.jcgt.org/published/0006/02/01/
//template <typename T>
__device__ bool intersect_aabb_branchless2(const glm::vec3& origin, const glm::vec3& direction, float& tmin) {
	constexpr glm::vec3 box_min = { 0, 0, 0 };
	constexpr glm::vec3 box_max = { grid_size, grid_size, grid_size };

	const glm::vec3 t1 = (box_min - origin) / direction;
	const glm::vec3 t2 = (box_max - origin) / direction;
	const glm::vec3 tMin = glm::min(t1, t2);
	const glm::vec3 tMax = glm::max(t1, t2);

	tmin = glm::max(glm::max(tMin.x, 0.f), glm::max(tMin.y, tMin.z));
	return glm::min(tMax.x, glm::min(tMax.y, tMax.z)) > tmin;
}

//Generate stratified sample of 2D [0,1]^2
__device__ glm::vec2 Random2DStratifiedSample(unsigned int& seed) {
	//Set the size of the pixel in stratums.
	constexpr int width2D = 4;
	constexpr int height2D = 4;
	constexpr float pixelWidth = 1.0f / width2D;
	constexpr float pixelHeight = 1.0f / height2D;

	const int chosenStratum = RandomIntBetween0AndMax(seed, width2D * height2D);
	//Compute stratum X in [0, width-1] and Y in [0,height -1]
	const int stratumX = chosenStratum % width2D;
	const int stratumY = (chosenStratum / width2D) % height2D;

	//Now we split up the pixel into [stratumX,stratumY] pieces.
	//Let's get the width and height of this sample

	const float stratumXStart = pixelWidth * stratumX;
	const float stratumYStart = pixelHeight * stratumY;

	const float randomPointInStratumX = stratumXStart + (RandomFloat(seed) * pixelWidth);
	const float randomPointInStratumY = stratumYStart + (RandomFloat(seed) * pixelHeight);
	return glm::vec2(randomPointInStratumX, randomPointInStratumY);
}

enum Refl_t { DIFF,
			  SPEC,
			  REFR,
			  PHONG,
			  LIGHT };

__device__ inline bool intersect_brick(glm::vec3 origin, const glm::vec3& direction, glm::vec3& normal, float& distance, Brick* brick) {
	origin.x = fmod(origin.x, 8.f);
	origin.y = fmod(origin.y, 8.f);
	origin.z = fmod(origin.z, 8.f);
	//poss = poss % 8;
	// Initialize
	glm::vec3 cb, tmax, tdelta;
	int stepX, outX, X = (int)origin.x;
	int stepY, outY, Y = (int)origin.y;
	int stepZ, outZ, Z = (int)origin.z;


	// Needed because sometimes the AABB intersect returns true while the ray is actually outside slightly. Only happens for faces that touch the AABB sides
	if (X >= cell_size || Y >= cell_size || Z >= cell_size) {
		//printf("mhmm positive\n");
		return false;
	}

	if (X < 0 || Y < 0 || Z < 0) {
		//printf("mhmm negative\n");
		return false;
	}


	if (direction.x > 0) {
		stepX = 1;
		outX = cell_size;
		cb.x = (X + 1);
	} else {
		stepX = -1;
		outX = -1;
		cb.x = X;
	}
	if (direction.y > 0.0f) {
		stepY = 1;
		outY = cell_size;
		cb.y = (Y + 1);
	} else {
		stepY = -1;
		outY = -1;
		cb.y = Y;
	}
	if (direction.z > 0.0f) {
		stepZ = 1;
		outZ = cell_size;
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
	distance = 0.f;

	// Stepping through grid
	while (1) {
		int sub_data = (X + Y * cell_size + Z * cell_size * cell_size) / 32;
		int bit = (X + Y * cell_size + Z * cell_size * cell_size) % 32;

		if (sub_data < 0 || sub_data > 15) {
			printf("uwu");
		}

		if (bit < 0 || bit > 31) {
			printf("OwO");
		}

		if (brick->data[sub_data] & (1 << bit)) {
			return true;
		}
	
		if (tmax.x < tmax.y) {
			if (tmax.x < tmax.z) {
				X += stepX;
				if (X == outX)
					return false;
				distance = tmax.x;
				tmax.x += tdelta.x;
				normal = glm::vec3(-stepX, 0, 0);
			} else {
				Z += stepZ;
				if (Z == outZ)
					return false;
				distance = tmax.z;
				tmax.z += tdelta.z;
				normal = glm::vec3(0, 0, -stepZ);
			}
		} else {
			if (tmax.y < tmax.z) {
				Y += stepY;
				if (Y == outY)
					return false;
				distance = tmax.y;
				tmax.y += tdelta.y;
				normal = glm::vec3(0, -stepY, 0);
			} else {
				Z += stepZ;
				if (Z == outZ)
					return false;
				distance = tmax.z;
				tmax.z += tdelta.z;
				normal = glm::vec3(0, 0, -stepZ);
			}
		}
	}
	return false;
}

__device__ inline bool intersect_voxel(glm::vec3 origin, const glm::vec3& direction, glm::vec3& normal, float& distance, Scene::GPUScene scene) {
	// Check if ray hits grid AABB
	float tminn;
	if (!intersect_aabb_branchless2(origin, direction, tminn)) {
		return false;
	}

	// Move ray to hitpoint
	if (tminn > 0) {
		origin += direction * tminn;

		constexpr glm::vec3 grid_center(grid_size / 2.f);
		glm::vec3 to_center = glm::abs(grid_center - origin);
		glm::vec3 signs = glm::sign(origin - grid_center);
		to_center /= glm::max(to_center.x, glm::max(to_center.y, to_center.z));
		normal = signs * glm::trunc(to_center + 0.000001f);

		origin += -normal * epsilon;
	}
	origin /= 8.f;

	// Initialize
	glm::vec3 cb, tmax, tdelta;
	int stepX, outX, X = ((int)origin.x);
	int stepY, outY, Y = ((int)origin.y);
	int stepZ, outZ, Z = ((int)origin.z);

	// Needed because sometimes the AABB intersect returns true while the ray is actually outside slightly. Only happens for faces that touch the AABB sides
	if (X < 0 || X >= cells || Y < 0 || Y >= cells || Z < 0 || Z >= cells) {
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
		outZ = cells;
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
	distance = 0.f;

	// Stepping through grid
	while (1) {
		Brick* brick = scene.grid[X + Y * cells + Z * cells * cells];
		if (brick != nullptr) {
			float sub_distance = 0.f;
			if (intersect_brick(origin * 8.f + direction * (distance * 8.f + epsilon), direction, normal, sub_distance, brick)) {
				distance += sub_distance + glm::max(tminn, 0.f) + epsilon;
				return true;
			}
			//return true;
		}

		constexpr int step_size = 1;

		if (tmax.x < tmax.y) {
			if (tmax.x < tmax.z) {
				X += stepX;
				if (X == outX)
					return false;
				distance = tmax.x;
				tmax.x += tdelta.x;
				normal = glm::vec3(-stepX, 0, 0);
			} else {
				Z += stepZ;
				if (Z == outZ)
					return false;
				distance = tmax.z;
				tmax.z += tdelta.z;
				normal = glm::vec3(0, 0, -stepZ);
			}
		} else {
			if (tmax.y < tmax.z) {
				Y += stepY;
				if (Y == outY)
					return false;
				distance = tmax.y;
				tmax.y += tdelta.y;
				normal = glm::vec3(0, -stepY, 0);
			} else {
				Z += stepZ;
				if (Z == outZ)
					return false;
				distance = tmax.z;
				tmax.z += tdelta.z;
				normal = glm::vec3(0, 0, -stepZ);
			}
		}
	}
	return false;
}

__device__ inline bool intersect_voxel_simple(const ShadowQueue& ray, Scene::GPUScene scene, glm::uvec3 grid_dimensions) {
	//// Check if ray hits grid AABB
	//float tminn;
	//if (!intersect_aabb_branchless2(ray, tminn)) {
	//	//if (!intersect_aabb_correct(ray, tminn)) {
	//	return false;
	//}

	//// Move ray to hitpoint
	//glm::vec3 origin = ray.origin;
	//if (tminn > 0) {
	//	origin += ray.direction * tminn;

	//	constexpr glm::vec3 grid_center(grid_size / 2.f);
	//	glm::vec3 to_center = glm::abs(grid_center - origin);
	//	glm::vec3 signs = glm::sign(origin - grid_center);
	//	to_center /= glm::max(to_center.x, glm::max(to_center.y, to_center.z));
	//	glm::vec3 normal = signs * glm::trunc(to_center + 0.000001f);

	//	origin += -normal * epsilon;
	//}

	//// Initialize
	//glm::vec3 cb, tmax, tdelta;
	//int stepX, outX, X = ((int)origin.x); // / 8;
	//int stepY, outY, Y = ((int)origin.y); // / 8;
	//int stepZ, outZ, Z = ((int)origin.z); // / 8;
	////origin /= 8.f;

	//// Needed because sometimes the AABB intersect returns true while the ray is actually outside slightly. Only happens for faces that touch the AABB sides
	//if (X < 0 || X >= grid_size || Y < 0 || Y >= grid_size || Z < 0 || Z >= grid_size) {
	//	//printf("full X: %i Y: %i Z: %i Bounce:%i Innie:%i\n", X, Y, Z, ray.bounces, innie);
	//	return false;
	//}

	//if (ray.direction.x > 0) {
	//	stepX = 1;
	//	outX = grid_size;
	//	cb.x = (X + 1);
	//} else {
	//	stepX = -1;
	//	outX = -1;
	//	cb.x = X;
	//}
	//if (ray.direction.y > 0.0f) {
	//	stepY = 1;
	//	outY = grid_size;
	//	cb.y = (Y + 1);
	//} else {
	//	stepY = -1, outY = -1;
	//	cb.y = Y;
	//}
	//if (ray.direction.z > 0.0f) {
	//	stepZ = 1;
	//	outZ = grid_size;
	//	cb.z = (Z + 1);
	//} else {
	//	stepZ = -1;
	//	outZ = -1;
	//	cb.z = Z;
	//}
	//float rxr, ryr, rzr;
	//if (ray.direction.x != 0) {
	//	rxr = 1.0f / ray.direction.x;
	//	tmax.x = (cb.x - origin.x) * rxr;
	//	tdelta.x = stepX * rxr;
	//} else
	//	tmax.x = 1000000;
	//if (ray.direction.y != 0) {
	//	ryr = 1.0f / ray.direction.y;
	//	tmax.y = (cb.y - origin.y) * ryr;
	//	tdelta.y = stepY * ryr;
	//} else
	//	tmax.y = 1000000;
	//if (ray.direction.z != 0) {
	//	rzr = 1.0f / ray.direction.z;
	//	tmax.z = (cb.z - origin.z) * rzr;
	//	tdelta.z = stepZ * rzr;
	//} else
	//	tmax.z = 1000000;

	//// Stepping through grid
	//while (1) {
	//	//if (scene.voxels[X + Y * grid_size + Z * grid_size * grid_size]) {
	//	//	//printf("bounces:%i x:%f y:%f z:%f\n", ray.bounces, origin.x, origin.y, origin.z);
	//	//	ray.distance = distance + glm::max(tminn, 0.f) + epsilon;
	//	//	return true;
	//	//}

	//	if (scene.grid[X / 8 + (Y / 8) * cells + (Z / 8) * cells * cells] != nullptr) {
	//		return true;
	//	}

	//	if (tmax.x < tmax.y) {
	//		if (tmax.x < tmax.z) {
	//			X += stepX;
	//			if (X == outX)
	//				return false;
	//			tmax.x += tdelta.x;
	//		} else {
	//			Z += stepZ;
	//			if (Z == outZ)
	//				return false;
	//			tmax.z += tdelta.z;
	//		}
	//	} else {
	//		if (tmax.y < tmax.z) {
	//			Y += stepY;
	//			if (Y == outY)
	//				return false;
	//			tmax.y += tdelta.y;
	//		} else {
	//			Z += stepZ;
	//			if (Z == outZ)
	//				return false;
	//			tmax.z += tdelta.z;
	//		}
	//	}
	//}
	return false;
}

//__device__ inline bool intersect_scene(RayQueue& ray, Scene::GPUScene sceneData) {
//	float d;
//	ray.distance = VERY_FAR;
//
//	for (int i = NUM_SPHERES; i--;) {
//		//d = spheres[i].intersect(ray);
//		if ((d = spheres[i].intersect(ray)) && d < ray.distance) {
//			ray.distance = d;
//			ray.identifier = i;
//			//ray.geometry_type = GeometryType::Sphere;
//		}
//	}
//
//	glm::vec3 normal;
//	if (intersect_voxel(ray, sceneData, normal)) {
//		ray.identifier = 0;
//		//ray.geometry_type == GeometryType::Triangle;
//		ray.distance = 1;
//		return true;
//	}
//
//
//	//if (sceneData.CUDACachedBVH.intersect(ray)) {
//	//	ray.geometry_type = GeometryType::Triangle;
//	//}
//	return ray.distance < VERY_FAR;
//}

//__device__ inline bool intersect_scene_simple(ShadowQueue& ray, Scene::GPUScene sceneData, const float& closestAllowed) {
//	float d;
//
//	/*if (sceneData.CUDACachedBVH.intersectSimple(ray, closestAllowed))
//		return true;*/
//
//	for (int i = NUM_SPHERES; i--;) {
//		if ((d = spheres[i].intersect_simple(ray)) && (d + epsilon) < closestAllowed) {
//			return true;
//		}
//	}
//	return false;
//}

/*
	Given a direction unit vector W, computes two other vectors U and V which 
	make uvw an orthonormal basis.
*/
//TODO(Dan): Implement Frisvad method.
__forceinline __device__ void computeOrthonormalBasisNaive(const glm::vec3& w, glm::vec3* u, glm::vec3* v) {
	if (fabs(w.x) > .9) { /*If W is to close to X axis then pick Y*/
		*u = glm::vec3{ 0.0f, 1.0f, 0.0f };
	} else { /*Pick X axis*/
		*u = glm::vec3{ 1.0f, 0.0f, 0.0f };
	}
	*u = normalize(cross(*u, w));
	*v = cross(w, *u);
}
__device__ glm::vec2 ConcentricSampleDisk(const glm::vec2& u) {
	//Map from [0,1] to [-1,1]
	glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1);

	// Handle degeneracy at the origin
	if (uOffset.x == 0 && uOffset.y == 0)
		return glm::vec2(0, 0);

	// Apply concentric mapping to point
	float theta, r;
	if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
		r = uOffset.x;
		theta = pi / 4 * (uOffset.y / uOffset.x);
	} else {
		r = uOffset.y;
		theta = pi / 2 - pi / 4 * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(std::cosf(theta), std::sinf(theta));
}

//Number of rays still active after the shade kernel.
__device__ unsigned int primary_ray_cnt = 0;
//The index of the ray at which we start generating more rays in ray generation step.
//Effectively is the last index which was previously generated + 1.
__device__ unsigned int start_position = 0;
//Ray number incremented by every thread in primary_rays ray generation
__device__ unsigned int raynr_primary = 0;
//Ray number to fetch different ray from every CUDA thread during the extend step.
__device__ unsigned int raynr_extend = 0;
//Ray number to fetch different ray from every CUDA thread in the shade step.
__device__ unsigned int raynr_shade = 0;
//Ray number to fetch different ray from every CUDA thread in the connect step.
__device__ unsigned int raynr_connect = 0;
//Number of shadow rays generated in shade step, which are placed in connect step.
__device__ unsigned int shadow_ray_cnt = 0;

///Kernel should be called after primary ray generation but before other wavefront steps.
__global__ void set_wavefront_globals() {

	//Get how many rays we created last generation step.
	const unsigned int progress_last_frame = ray_queue_buffer_size - primary_ray_cnt;

	//The starting position for the next step is where we left off last time.
	//Last step we progressed from the start_position by progress_last_frame rays.
	//Next step we start from prev starting position incremented by how much we progressed this frame
	start_position += progress_last_frame;
	start_position = start_position % (render_width * render_height);
	//Zero out counters atomically incremented for all wavefront kernels.
	shadow_ray_cnt = 0;
	primary_ray_cnt = 0;
	raynr_primary = 0;
	raynr_extend = 0;
	raynr_shade = 0;
	raynr_connect = 0;
}

/// Generate primary rays. Fill ray_buffer up till max length.
__global__ void primary_rays(RayQueue* ray_buffer, glm::vec3 camera_right, glm::vec3 camera_up, glm::vec3 camera_direction, glm::vec3 O, unsigned int frame, float focalDistance, float lens_radius, Scene::GPUScene sceneData, glm::vec4* blit_buffer) {

	//Fill ray buffer up to ray_queue_buffer_size.
	while (true) {
		const unsigned int index = atomicAdd(&raynr_primary, 1);
		//Buffer already includes rays generated by previous "shade" step (primary_ray_cnt)
		const unsigned int ray_index_buffer = index + primary_ray_cnt;
		if (ray_index_buffer > ray_queue_buffer_size - 1) {
			return;
		}
		//Initialize random seed
		unsigned int seed = (frame * 147565741) * 720898027 * index;

		//Compute (x,y) coords based on position in buffer.
		// X goes (left -> right); Y goes (top -> bottom)

		const unsigned int x = (start_position + index) % render_width;
		const unsigned int y = ((start_position + index) / render_width) % render_height;

		//Get random stratified points inside pixel;
		glm::vec2 sample2D = Random2DStratifiedSample(seed);
		const float rand_point_pixelX = x - sample2D.x;
		const float rand_point_pixelY = y - sample2D.y;

#if 0 //Ordinary random points
		const float rand_point_pixelX = x - RandomFloat(seed);
		const float rand_point_pixelY = y - RandomFloat(seed);
#endif

		const float normalized_i = (rand_point_pixelX / (float)render_width) - 0.5f;
		const float normalized_j = ((render_height - rand_point_pixelY) / (float)render_height) - 0.5f;

		//Normal direction which we would compute even without DoF
		glm::vec3 directionToFocalPlane = camera_direction + normalized_i * camera_right + normalized_j * camera_up;
		directionToFocalPlane = glm::normalize(directionToFocalPlane);

		//Get the convergence point which is at focalDistance)
		//TODO(Dan): I currently multiply by 3 because I felt it would be easier for the ImGui slider.
		// Fix this by modifying how slider works?
		const int ImGui_slider_hack = 3.0f;
		glm::vec3 convergencePoint = O + focalDistance * ImGui_slider_hack * directionToFocalPlane;

		glm::vec2 lens_sample(RandomFloat(seed), RandomFloat(seed));
		glm::vec2 pLens = lens_radius * ConcentricSampleDisk(lens_sample);
		glm::vec3 newOrigin = O + camera_right * pLens.x + camera_up * pLens.y;

		glm::vec3 direction = glm::normalize(convergencePoint - newOrigin);

		ray_buffer[ray_index_buffer] = { newOrigin, direction, { 1, 1, 1 }, { 0, 0, 0 }, 0, 0, 0, y * render_width + x };
	}
}

/// Advance the ray segments once
__global__ void __launch_bounds__(128, 8) extend(RayQueue* ray_buffer, Scene::GPUScene sceneData, glm::vec4* blit_buffer, unsigned int seed) {
	while (true) {
		const unsigned int index = atomicAdd(&raynr_extend, 1);

		if (index > ray_queue_buffer_size - 1) {
			return;
		}
		RayQueue& ray = ray_buffer[index];

		ray.distance = VERY_FAR;
		//intersect_voxel(ray, sceneData, ray.normal, {32, 32, 32});

		if (intersect_voxel(ray.origin, ray.direction, ray.normal, ray.distance, sceneData)) {
			//glm::vec3 yoyo = ray.origin + ray.direction * ray.distance;
			//atomicAdd(&blit_buffer[ray.pixel_index].r, 1.f);
			//atomicAdd(&blit_buffer[ray.pixel_index].g, 1.f);
			//atomicAdd(&blit_buffer[ray.pixel_index].b, 1.f);
			atomicAdd(&blit_buffer[ray.pixel_index].r, ray.normal.x);
			atomicAdd(&blit_buffer[ray.pixel_index].g, ray.normal.y);
			atomicAdd(&blit_buffer[ray.pixel_index].b, ray.normal.z);
			atomicAdd(&blit_buffer[ray.pixel_index].a, 1.f);
			//printf("x:%f y:%f z:%f\n", ray.normal.x, ray.normal.y, ray.normal.z);
		}
	}
}

/// Process collisions and spawn extension and shadow rays.
/// Rays that continue get placed in ray_buffer_next to be processed next frame
__global__ void __launch_bounds__(128, 8) shade(RayQueue* ray_buffer, RayQueue* ray_buffer_next, ShadowQueue* shadowQueue, Scene::GPUScene sceneData, glm::vec4* blit_buffer, unsigned int frame) {

	while (true) {
		const unsigned int index = atomicAdd(&raynr_shade, 1);

		if (index > ray_queue_buffer_size - 1) {
			return;
		}

		int new_frame = 0;
		RayQueue& ray = ray_buffer[index];

		//Each iteration we add color to the blit_buffer.
		//Color can be non-zero if sun/sky or we're counting emisivity for different objects.
		glm::vec3 color = glm::vec3(0.f);
		glm::vec3 object_color;
		unsigned int seed = (frame * ray.pixel_index * 147565741) * 720898027 * index;
		int reflection_type = DIFF;

		if (ray.distance < VERY_FAR) {
			ray.origin += ray.direction * ray.distance;
			//Prevent self-intersection
			ray.origin += ray.normal * 2.f * epsilon;

			// Generate new shadow ray
			glm::vec3 sunSampleDir = getConeSample(sunDirection, 1.0f - sunAngularDiameterCos, seed);
			float sunLight = dot(ray.normal, sunSampleDir);

			// < 0.f means sun is behind the surface
			if (sunLight > 0.f) {
				unsigned shadow_index = atomicAdd(&shadow_ray_cnt, 1);
				shadowQueue[shadow_index] = { ray.origin, sunSampleDir, 2.0f * ray.direct * (sun(sunSampleDir) * sunLight * 1E-5f), ray.pixel_index };
			}

			if (ray.bounces < MAX_BOUNCES) {
#if 0 // Stratified sampling.
				glm::vec2 samples = Random2DStratifiedSample(seed); 
				float r1 = 2.f * pi * samples.x;
				float r2 = samples.y;
#else
				float r1 = 2.f * pi * RandomFloat(seed);
				float r2 = RandomFloat(seed);
#endif
				float r2s = sqrt(r2);

				// Transform to hemisphere coordinate system
				glm::vec3 u, v;
				computeOrthonormalBasisNaive(ray.normal, &u, &v);
				// Get sample on hemisphere
				ray.direction = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + ray.normal * sqrt(1 - r2));
			}

			//Russian roullete
			float p = glm::min(1.0f, glm::max(ray.direct.z, glm::max(ray.direct.x, ray.direct.y)));
			if (ray.bounces < MAX_BOUNCES && p > (0 + epsilon) && RandomFloat(seed) <= p) {
				//Add rays into the next ray_buffer to be processed next frame
				ray.bounces++;
				ray.direct *= 1.0f / p;

				unsigned primary_index = atomicAdd(&primary_ray_cnt, 1);
				ray_buffer_next[primary_index] = ray;
			} else { // MAX BOUNCES
				new_frame++;
			}

		} else { //NOTHING HIT
			// Don't generate new extended ray. Directly add emmisivity of sun/sky.
			color += ray.direct * sunsky(ray.direction);
			new_frame++;
		}

		//Color is added every frame to buffer. However color can only be non-zero for sun/sky and if emmisive surface
		//was hit.
		//TODO(Dan): Perf increase if only add when != 0? How to interact with sky = black?
		atomicAdd(&blit_buffer[ray.pixel_index].r, color.r);
		atomicAdd(&blit_buffer[ray.pixel_index].g, color.g);
		atomicAdd(&blit_buffer[ray.pixel_index].b, color.b);
		atomicAdd(&blit_buffer[ray.pixel_index].a, new_frame);
	}
}

/// Proccess shadow rays
__global__ void __launch_bounds__(128, 8) connect(ShadowQueue* queue, Scene::GPUScene sceneData, glm::vec4* blit_buffer) {
	while (true) {
		const unsigned int index = atomicAdd(&raynr_connect, 1);

		if (index >= shadow_ray_cnt) {
			return;
		}

		ShadowQueue& ray = queue[index];

		if (!intersect_voxel_simple(ray, sceneData, { 32, 32, 32 })) {
			atomicAdd(&blit_buffer[ray.pixel_index].r, ray.color.r);
			atomicAdd(&blit_buffer[ray.pixel_index].g, ray.color.g);
			atomicAdd(&blit_buffer[ray.pixel_index].b, ray.color.b);
		}
	}
}

__global__ void blit_onto_framebuffer(glm::vec4* blit_buffer) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= render_width || y >= render_height) {
		return;
	}

	const int index = y * render_width + x;
	glm::vec4 color = blit_buffer[index];
	glm::vec4 cl = glm::vec4(color.r, color.g, color.b, 1) / color.a;
	cl.a = 1;

	surf2Dwrite<glm::vec4>(cl, surf, x * sizeof(glm::vec4), y, cudaBoundaryModeZero);
	//surf2Dwrite<glm::vec4>(glm::pow(cl, glm::vec4(1.0f / 2.2f)), surf, x * sizeof(glm::vec4), y, cudaBoundaryModeZero);
}

cudaError launch_kernels(cudaArray_const_t array, glm::vec4* blit_buffer, Scene::GPUScene sceneData, RayQueue* ray_buffer, RayQueue* ray_buffer_next, ShadowQueue* shadow_queue) {
	static bool first_time = true;
	static bool reset_buffer = false;
	static unsigned int frame = 1;

	if (first_time) {
		first_time = false;

		/*	Sphere sphere_data[NUM_SPHERES] = { { 16.5, { 0, 40, 16.5f }, { 1, 1, 1 }, { 0, 0, 0 }, DIFF },
											{ 16.5, { 40, 0, 16.5f }, { 0.5, 0.5, 0.06 }, { 0, 0, 0 }, DIFF },
											{ 16.5, { -40, -50, 36.5f }, { 0.6, 0.5, 0.4 }, { 0, 0, 0 }, DIFF },
											{ 16.5, { -40, -50, 16.5f }, { 0.6, 0.5, 0.4 }, { 0, 0, 0 }, DIFF },
											{ 1e4f, { 0, 0, -1e4f - 20 }, { 1, 1, 1 }, { 0, 0, 0 }, DIFF },
											{ 20, { 0, -80, 20 }, { 1.0, 0.0, 0.0 }, { 0, 0, 0 }, DIFF },
											{ 9, { 0, -80, 120.0f }, { 0.0, 1.0, 0.0 }, { 3, 3, 3 }, LIGHT } };
		cudaMemcpyToSymbol(spheres, sphere_data, NUM_SPHERES * sizeof(Sphere));*/

		float sun_angular = cos(sunSize * pi / 180.f);
		cuda(MemcpyToSymbol(sunAngularDiameterCos, &sun_angular, sizeof(float)));
	}

	cudaError cuda_err;
	static glm::vec3 last_pos;
	static glm::vec3 last_dir;
	static float last_focaldistance = 1;
	static float last_lensradius = 0.02f;

	cuda_err = cuda(BindSurfaceToArray(surf, array));

	if (cuda_err) {
		return cuda_err;
	}

	const glm::vec3 camera_right = glm::normalize(glm::cross(camera.direction, camera.up)) * 1.5f * ((float)render_width / render_height);
	const glm::vec3 camera_up = glm::normalize(glm::cross(camera_right, camera.direction)) * 1.5f;

	reset_buffer = last_pos != camera.position || last_dir != camera.direction || last_focaldistance != camera.focalDistance || camera.lensRadius != last_lensradius;

	if (sun_position_changed) {
		sun_position_changed = false;
		reset_buffer = true;
		cuda(MemcpyToSymbol(SunPos, &sun_position, sizeof(glm::vec2)));
		glm::vec3 sun_direction = glm::normalize(fromSpherical((sun_position - glm::vec2(0.0, 0.5)) * glm::vec2(6.28f, 3.14f)));
		cuda(MemcpyToSymbol(sunDirection, &sun_direction, sizeof(glm::vec3)));
	}

	if (reset_buffer) {
		reset_buffer = false;
		cudaMemset(blit_buffer, 0, render_width * render_height * sizeof(float4));

		int new_value = 0;
		cuda(MemcpyToSymbol(primary_ray_cnt, &new_value, sizeof(int)));
	}
	primary_rays<<<sm_cores * 8, 128>>>(ray_buffer, camera_right, camera_up, camera.direction, camera.position, frame, camera.focalDistance, camera.lensRadius, sceneData, blit_buffer);
	set_wavefront_globals<<<1, 1>>>();
	extend<<<sm_cores * 8, 128>>>(ray_buffer, sceneData, blit_buffer, frame);
	//shade<<<sm_cores * 8, 128>>>(ray_buffer, ray_buffer_next, shadow_queue, sceneData, blit_buffer, frame);
	//connect<<<sm_cores * 8, 128>>>(shadow_queue, sceneData, blit_buffer);

	dim3 threads = dim3(16, 16, 1);
	dim3 blocks = dim3(render_width / threads.x, render_height / threads.y, 1);
	blit_onto_framebuffer<<<blocks, threads>>>(blit_buffer);

	cuda(DeviceSynchronize());

	//Frame is used as XORSHIFT seed, but we must ensure it's not 0
	if (frame == UINT_MAX)
		frame = 0;

	frame++;

	//hold_frame++;
	last_pos = camera.position;
	last_dir = camera.direction;
	last_focaldistance = camera.focalDistance;
	last_lensradius = camera.lensRadius;

	return cudaSuccess;
}