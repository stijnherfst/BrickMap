#include "stdafx.h"
#include "sunsky.cuh"

#include "assert_cuda.h"
#include "cuda_surface_types.h"
#include "device_launch_parameters.h"
#include "surface_functions.h"

#include "cuda_definitions.h"

//Define BVH_DEBUG to zero to output only what the BVH looks like
#define BVH_DEBUG 0

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

inline __host__ __device__ float dot(const glm::vec4& v1, const glm::vec3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

struct Sphere {
	float radius;
	glm::vec3 position, color;
	glm::vec3 emmission;
	Refl_t refl;

	__device__ float intersect(const RayQueue& r) const {
		glm::vec3 op = position - r.origin;
		float t;
		float b = glm::dot(op, r.direction);
		float disc = b * b - dot(op, op) + radius * radius;
		if (disc < 0)
			return 0;

		disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}

	__device__ float intersect_simple(const ShadowQueue& r) const {
		glm::vec3 op = position - r.origin;
		float t;
		float b = glm::dot(op, r.direction);
		float disc = b * b - dot(op, op) + radius * radius;
		if (disc < 0)
			return 0;

		disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}

	__device__ glm::vec3 random_point(unsigned int& seed) const {
		float u = RandomFloat(seed);
		float v = RandomFloat(seed);

		float cosPhi = 2.0f * u - 1.0f;
		float sinPhi = sqrt(1.0f - cosPhi * cosPhi);
		float theta = 2 * pi * v;

		float x = radius * sinPhi * sin(theta);
		float y = radius * cosPhi;
		float z = radius * sinPhi * cos(theta);

		return position + glm::vec3(x, y, z);
	}
};

__constant__ Sphere spheres[NUM_SPHERES];

__device__ inline bool intersect_voxel(RayQueue& ray, Scene::GPUScene scene) {
	glm::vec3 raydir, curpos;
	curpos = ray.origin;
	raydir = ray.direction;

	// setup 3DDDA (double check reusability of primary ray data)
	glm::vec3 cb, tmax, tdelta, cell;
	cell = curpos;
	int stepX, outX, X = (int)cell.x;
	int stepY, outY, Y = (int)cell.y;
	int stepZ, outZ, Z = (int)cell.z;

	if ((X < 0) || (X >= grid_size) || (Y < 0) || (Y >= grid_size) || (Z < 0) || (Z >= grid_size))
		return false;
	if (raydir.x > 0) {
		stepX = 1, outX = grid_size;
		cb.x = (X + 1);
	} else {
		stepX = -1, outX = -1;
		cb.x = X;
	}
	if (raydir.y > 0.0f) {
		stepY = 1, outY = grid_size;
		cb.y = (Y + 1);
	} else {
		stepY = -1, outY = -1;
		cb.y = Y;
	}
	if (raydir.z > 0.0f) {
		stepZ = 1, outZ = grid_size;
		cb.z = (Z + 1);
	} else {
		stepZ = -1, outZ = -1;
		cb.z = Z;
	}
	float rxr, ryr, rzr;
	if (raydir.x != 0) {
		rxr = 1.0f / raydir.x;
		tmax.x = (cb.x - curpos.x) * rxr;
		tdelta.x = stepX * rxr;
	} else
		tmax.x = 1000000;
	if (raydir.y != 0) {
		ryr = 1.0f / raydir.y;
		tmax.y = (cb.y - curpos.y) * ryr;
		tdelta.y = stepY * ryr;
	} else
		tmax.y = 1000000;
	if (raydir.z != 0) {
		rzr = 1.0f / raydir.z;
		tmax.z = (cb.z - curpos.z) * rzr;
		tdelta.z = stepZ * rzr;
	} else
		tmax.z = 1000000;
	// start stepping

	//return true;
	// trace primary ray
	while (1) {
		//if (X > 16 && Y > 16 && X < 32 && Y < 32)
		//	return true;
		if (scene.voxels[X + Y * grid_size + Z * grid_size * grid_size]) {
			//printf("a");
			return true;
		}
		if (tmax.x < tmax.y) {
			if (tmax.x < tmax.z) {
				X = X + stepX;
				if (X == outX)
					return false;
				tmax.x += tdelta.x;
			} else {
				Z = Z + stepZ;
				if (Z == outZ)
					return false;
				tmax.z += tdelta.z;
			}
		} else {
			if (tmax.y < tmax.z) {
				Y = Y + stepY;
				if (Y == outY)
					return false;
				tmax.y += tdelta.y;
			} else {
				Z = Z + stepZ;
				if (Z == outZ)
					return false;
				tmax.z += tdelta.z;
			}
		}
	}

//testloop:
//	while (1) {
//		/*list = grid[X + (Y << GRIDSHFT) + (Z << (GRIDSHFT * 2))];
//		while (list) {
//			Primitive* pr = list->GetPrimitive();
//			int result;
//			if (pr->GetLastRayID() != a_Ray.GetID())
//				if (result = pr->Intersect(a_Ray, a_Dist)) {
//					a_Prim = pr;
//					retval = result;
//				}
//			list = list->GetNext();
//		}*/
//		if (scene.voxels[X + Y * 264 + Z * 264 * 264]) {
//			return true;
//		}
//
//		if (tmax.x < tmax.y) {
//			if (tmax.x < tmax.z) {
//				if (a_Dist < tmax.x)
//					break;
//				X = X + stepX;
//				if (X == outX)
//					break;
//				tmax.x += tdelta.x;
//			} else {
//				if (a_Dist < tmax.z)
//					break;
//				Z = Z + stepZ;
//				if (Z == outZ)
//					break;
//				tmax.z += tdelta.z;
//			}
//		} else {
//			if (tmax.y < tmax.z) {
//				if (a_Dist < tmax.y)
//					break;
//				Y = Y + stepY;
//				if (Y == outY)
//					break;
//				tmax.y += tdelta.y;
//			} else {
//				if (a_Dist < tmax.z)
//					break;
//				Z = Z + stepZ;
//				if (Z == outZ)
//					break;
//				tmax.z += tdelta.z;
//			}
//		}
//	}
	return false;
}

__device__ inline bool intersect_scene(RayQueue& ray, Scene::GPUScene sceneData) {
	float d;
	ray.distance = VERY_FAR;

	for (int i = NUM_SPHERES; i--;) {
		//d = spheres[i].intersect(ray);
		if ((d = spheres[i].intersect(ray)) && d < ray.distance) {
			ray.distance = d;
			ray.identifier = i;
			ray.geometry_type = GeometryType::Sphere;
		}
	}


	if (intersect_voxel(ray, sceneData)) {
		ray.identifier = 0;
		ray.geometry_type == GeometryType::Triangle;
		ray.distance = 1;
		return true;
	}


	//if (sceneData.CUDACachedBVH.intersect(ray)) {
	//	ray.geometry_type = GeometryType::Triangle;
	//}
	return ray.distance < VERY_FAR;
}

__device__ inline bool intersect_scene_DEBUG(RayQueue& ray, Scene::GPUScene sceneData, int* traversals) {
	float d;
	ray.distance = VERY_FAR;

	for (int i = NUM_SPHERES; i--;) {
		//d = spheres[i].intersect(ray);
		if ((d = spheres[i].intersect(ray)) && d < ray.distance) {
			ray.distance = d;
			ray.identifier = i;
			ray.geometry_type = GeometryType::Sphere;
		}
	}

	//if (sceneData.CUDACachedBVH.intersect_debug(ray, traversals)) {
	//	ray.geometry_type = GeometryType::Triangle;
	//}
	return ray.distance < VERY_FAR;
}

__device__ inline bool intersect_scene_simple(ShadowQueue& ray, Scene::GPUScene sceneData, const float& closestAllowed) {
	float d;

	/*if (sceneData.CUDACachedBVH.intersectSimple(ray, closestAllowed))
		return true;*/

	for (int i = NUM_SPHERES; i--;) {
		if ((d = spheres[i].intersect_simple(ray)) && (d + epsilon) < closestAllowed) {
			return true;
		}
	}
	return false;
}

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

		const int x = (start_position + index) % render_width;
		const int y = ((start_position + index) / render_width) % render_height;

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

		//ray_buffer[ray_index_buffer] 
		RayQueue ray = { newOrigin, direction, { 1, 1, 1 }, 0, 0, 0, y * render_width + x };

		ray.distance = VERY_FAR;
	//intersect_scene(ray, sceneData);
		if (intersect_voxel(ray, sceneData)) {
			atomicAdd(&blit_buffer[ray.index].r, 1.f);
			atomicAdd(&blit_buffer[ray.index].g, 0.f);
			atomicAdd(&blit_buffer[ray.index].b, 1.f);
			atomicAdd(&blit_buffer[ray.index].a, 1.f);
		}
	}
}

/// Execute "extend" step but compute ray color based on amount of BVH traversal steps and write to blit_buffer.
__global__ void __launch_bounds__(128, 8) extend_debug_BVH(RayQueue* ray_buffer, Scene::GPUScene sceneData, glm::vec4* blit_buffer) {
	while (true) {
		unsigned int index = atomicAdd(&raynr_extend, 1);

		if (index > ray_queue_buffer_size - 1) {
			return;
		}

		RayQueue& ray = ray_buffer[index];

		ray.distance = VERY_FAR;
		int traversals = 0;
		intersect_scene_DEBUG(ray, sceneData, &traversals);

		//Determine colors

		glm::vec3 color = {};
		int green = (0.0002f * traversals) * 255.99f;
		green = green > 255 ? 255 : green;
		blit_buffer[ray.index].g = green;
		blit_buffer[ray.index].a = 1;

		if (traversals >= 70) { //Color very costly regions distinctly
			int red = green;
			blit_buffer[ray.index].r = red;
			blit_buffer[ray.index].g = 0;
		}
	}
}

/// Advance the ray segments once
__global__ void __launch_bounds__(128, 8) extend(RayQueue* ray_buffer, Scene::GPUScene sceneData) {
	while (true) {
		const unsigned int index = atomicAdd(&raynr_extend, 1);

		if (index > ray_queue_buffer_size - 1) {
			return;
		}
		RayQueue& ray = ray_buffer[index];

		ray.distance = VERY_FAR;
		intersect_scene(ray, sceneData);
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
		unsigned int seed = (frame * ray.index * 147565741) * 720898027 * index;
		int reflection_type = DIFF;

		if (ray.distance < VERY_FAR) {


			ray.origin += ray.direction * ray.distance;

			glm::vec3 normal;
			if (ray.geometry_type == GeometryType::Sphere) {
				const Sphere& object = spheres[ray.identifier];
				normal = (ray.origin - object.position) / object.radius;
				reflection_type = object.refl;

				if (reflection_type != LIGHT) {
					ray.direct *= object.color;
				}
				object_color = object.color;
			} else {
				//normal = glm::normalize(glm::cross(triangle.e1, triangle.e2));
				//reflection_type = DIFF;
				object_color = glm::vec3(1.f, 0.f, 1.f);
			}

			atomicAdd(&blit_buffer[ray.index].r, object_color.x);
			atomicAdd(&blit_buffer[ray.index].g, object_color.y);
			atomicAdd(&blit_buffer[ray.index].b, object_color.z);
			atomicAdd(&blit_buffer[ray.index].a, 1);

			return;

			bool outside = dot(normal, ray.direction) < 0;
			normal = outside ? normal : normal * -1.f; // make n front facing is we are inside an object

			//Prevent self-intersection
			ray.origin += normal * epsilon;

			//Handle light case before resetting previous ray specular
			if (reflection_type == LIGHT) {
				color = glm::vec3(0.0f);
				ray.direct = glm::vec3(0.0, 0.0f, 0.0f);
			}

			switch (reflection_type) {
				case LIGHT: {
					break;
				}
				case DIFF: {
					// Generate new shadow ray
					glm::vec3 sunSampleDir = getConeSample(sunDirection, 1.0f - sunAngularDiameterCos, seed);
					float sunLight = dot(normal, sunSampleDir);

					// < 0.f means sun is behind the surface
					if (RandomFloat(seed) < 0.5f) { // NEE sample the sun
						if (sunLight > 0.f) {
							unsigned shadow_index = atomicAdd(&shadow_ray_cnt, 1);
							shadowQueue[shadow_index] = { ray.origin, sunSampleDir, 2.0f * ray.direct * (sun(sunSampleDir) * sunLight * 1E-5f), ray.index };
						}
					} else { // NEE Sample the light source in the scene
						//TODO(Dan): Hardcoded spheres[6] as only light source. Use light array.
						Sphere& lightsource = spheres[6];

						float cosPhi = 2.0f * RandomFloat(seed) - 1.0f;
						float sinPhi = std::sqrt(1.0f - cosPhi * cosPhi);
						float theta = 2.0f * pi * RandomFloat(seed);

						float x = lightsource.position.x + lightsource.radius * sinPhi * std::sinf(theta);
						float y = lightsource.position.y + lightsource.radius * cosPhi;
						float z = lightsource.position.z + lightsource.radius * sinPhi * std::cosf(theta);

						glm::vec3 lightVector = glm::vec3(x, y, z) - ray.origin;
						glm::vec3 nL = glm::normalize(glm::vec3(x, y, z) - lightsource.position);
						glm::vec3 lightDir = glm::normalize(lightVector);
						float cosSurfaceToLight = glm::dot(normal, lightDir);
						float cosLightToSurface = glm::dot(nL, -lightDir);

						if (cosSurfaceToLight > 0 && cosLightToSurface > 0) {

							float closestAllowed = glm::length(lightVector);
							float area = 4 * pi * lightsource.radius * lightsource.radius;
							float solidAngle = (cosLightToSurface * area) / glm::dot(lightVector, lightVector);
							glm::vec3 shadowColor = lightsource.emmission * 2.0f * ray.direct * solidAngle * inv_pi * cosSurfaceToLight;

							unsigned shadow_index = atomicAdd(&shadow_ray_cnt, 1);

							shadowQueue[shadow_index] = { ray.origin, lightDir, shadowColor, ray.index, closestAllowed };
						}
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
						computeOrthonormalBasisNaive(normal, &u, &v);
						// Get sample on hemisphere
						const glm::vec3 d = glm::normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + normal * sqrt(1 - r2));
						ray.direction = d;
					}

					break;
				}
			}

			//Russian roullete
			float p = glm::min(1.0f, glm::max(ray.direct.z, glm::max(ray.direct.x, ray.direct.y)));
			//float p = 1.0f;
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
			// Don't generate new extended ray. Directly add emisivity of sun/sky.
			color += (ray.lastSpecular == false) ? ray.direct * sky(ray.direction) : ray.direct * sunsky(ray.direction);
			new_frame++;
		}

		//Color is added every frame to buffer. However color can only be non-zero for sun/sky and if emmisive surface
		//was hit.
		//TODO(Dan): Perf increase if only add when != 0? How to interact with sky = black?
		atomicAdd(&blit_buffer[ray.index].r, color.r);
		atomicAdd(&blit_buffer[ray.index].g, color.g);
		atomicAdd(&blit_buffer[ray.index].b, color.b);
		atomicAdd(&blit_buffer[ray.index].a, new_frame);
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

		if (!intersect_scene_simple(ray, sceneData, ray.closestDistance)) {
			atomicAdd(&blit_buffer[ray.buffer_index].r, ray.color.r);
			atomicAdd(&blit_buffer[ray.buffer_index].g, ray.color.g);
			atomicAdd(&blit_buffer[ray.buffer_index].b, ray.color.b);
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

	surf2Dwrite<glm::vec4>(glm::pow(cl / (cl + 1.f), glm::vec4(1.0f / 2.2f)), surf, x * sizeof(glm::vec4), y, cudaBoundaryModeZero);
}

cudaError launch_kernels(cudaArray_const_t array, glm::vec4* blit_buffer, Scene::GPUScene sceneData, RayQueue* ray_buffer, RayQueue* ray_buffer_next, ShadowQueue* shadow_queue) {
	static bool first_time = true;
	static bool reset_buffer = false;
	static unsigned int frame = 1;

	if (first_time) {
		first_time = false;

		Sphere sphere_data[NUM_SPHERES] = { { 16.5, { 0, 40, 16.5f }, { 1, 1, 1 }, { 0, 0, 0 }, DIFF },
											{ 16.5, { 40, 0, 16.5f }, { 0.5, 0.5, 0.06 }, { 0, 0, 0 }, DIFF },
											{ 16.5, { -40, -50, 36.5f }, { 0.6, 0.5, 0.4 }, { 0, 0, 0 }, DIFF },
											{ 16.5, { -40, -50, 16.5f }, { 0.6, 0.5, 0.4 }, { 0, 0, 0 }, DIFF },
											{ 1e4f, { 0, 0, -1e4f - 20 }, { 1, 1, 1 }, { 0, 0, 0 }, DIFF },
											{ 20, { 0, -80, 20 }, { 1.0, 0.0, 0.0 }, { 0, 0, 0 }, DIFF },
											{ 9, { 0, -80, 120.0f }, { 0.0, 1.0, 0.0 }, { 3, 3, 3 }, LIGHT } };
		cudaMemcpyToSymbol(spheres, sphere_data, NUM_SPHERES * sizeof(Sphere));

		float sun_angular = cos(sunSize * pi / 180.f);
		cuda(MemcpyToSymbol(sunAngularDiameterCos, &sun_angular, sizeof(float)));
	}

	cudaError cuda_err;
	static glm::vec3 last_pos;
	static glm::vec3 last_dir;
	static float last_focaldistance = 1;
	static float last_lensradius = 0.02;

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
#if BVH_DEBUG
	extend_debug_BVH<<<40, 128>>>(ray_buffer, sceneData, blit_buffer);
#else
	//extend<<<sm_cores * 8, 128>>>(ray_buffer, sceneData);
	//shade<<<sm_cores * 8, 128>>>(ray_buffer, ray_buffer_next, shadow_queue, sceneData, blit_buffer, frame);
	//connect<<<sm_cores * 8, 128>>>(shadow_queue, sceneData, blit_buffer);
#endif

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