// This code heavily used PBRT as a reference, for which the copyright is:

/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.
    This file is part of pbrt.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
The changes made to original PBRT code include:
-Removed all the intermediary steps involved in building a BVH. 
We now directly compute the cache-friendly BVH which results in quite some speedup
in build since we're not even allocating memory and also  jumping through 32 byte nodes is faster
-Streamlining certain operations/data that were general in PBRT. Some of which:
PrimitiveInfo, Primitives only as triangles. Placing into buckets & SAH code is simpler with less edge cases
for number of primitives, memory arena was removed
-Code is heavily commented, which may prove useful for future implementers.
-More explicit parameters you can tweak for SAH. Like:
Cost per node traversal and cost per triangle intersection, bucketNumber, max primitives in a node

Thus it is now less general, but simpler and faster to build.

*/
#pragma once

enum class PartitionAlgorithm { Middle,
								EqualCounts,
								SAH };

class BVH {
public:
	BVH(std::vector<Triangle>& primitives, std::vector<BBox> primitivesBBoxes,
		PartitionAlgorithm partitionAlgo);
	~BVH() = default;

	struct BVHNode {
		BBox bbox;
		union {
			int primitiveOffset;
			int secondChildOffset;
		};
		uint16_t primitiveCount;
		uint8_t splitAxis;
		char pad[1];

		void initLeaf(int first, int n, const BBox& box);
		void initInterior(int axis, const BVHNode& left, const BVHNode& right);
	};
	static_assert(sizeof(BVHNode) == 32, "Size is not correct");

	std::vector<BVHNode> nodes;
	int nNodes = 0;
	const PartitionAlgorithm partitionAlgorithm = PartitionAlgorithm::SAH;

private:
	//Number of buckets to split up axis for SAH
	static constexpr int bucket_number = 14;
	//Maximum number of primitives allowed in BVH leaf node for SAH
	static constexpr int max_prim_number = 4;

	//cost to traverse a node in the BVH. If INTERSECTION_COST is 1 then this is just percentage slower/faster
	static constexpr float TRAVERSAL_COST = 1.0f;

	//cost to intersect a triangle int the BVH. Leave this as "1" and change TRAVERSAL_COST instead
	static constexpr float INTERSECTION_COST = 1.0f;

	static_assert(INTERSECTION_COST == 1, "You can vary traversal_cost instead of intersection_cost");
	static_assert(bucket_number >= 2, "Buckets should be enough to split the space! At least 2 required");

	struct PrimitiveInfo {
		uint32_t primitiveNumber = {};
		BBox bbox = {};
		glm::vec3 centroid = {};
		PrimitiveInfo() = default;
		PrimitiveInfo(uint32_t primitiveNumber, BBox& bbox)
			: primitiveNumber(primitiveNumber)
			, bbox(bbox)
			, centroid(bbox.bounds[0] * 0.5f + bbox.bounds[1] * 0.5f) {}
	};

	//Calculate the bucket in which a primitive is to be placed for the SAH algorithm.
	//Bucket is computed by evenly splitting along a dimension the intervals in the centroid bounding box
	int computeBucket(PrimitiveInfo primitive, glm::vec3 centroidBottom, glm::vec3 centroidTop, int dim);

	void recursiveBuild(int start, int end, int* nNodes,
						std::vector<PrimitiveInfo>& primitiveInfo,
						std::vector<Triangle>& orderedPrimitives,
						const std::vector<Triangle>& primitives);
};

//Class meant to store a cached BVH to send to the GPU
class CachedBVH {
public:
	CachedBVH() = default;

	BVH::BVHNode* nodes = nullptr;
	Triangle* primitives = nullptr;

	__device__ bool intersect(RayQueue& ray) {
		bool hit = false;
		glm::vec3 invDir = 1.f / ray.direction;
		int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
		// Follow ray through BVH nodes to find primitive intersections
		int toVisitOffset = 0, currentNodeIndex = 0;
		int nodesToVisit[64];
		while (true) {
			const auto* node = &nodes[currentNodeIndex];
			// Check ray against BVH node
			if (node->bbox.intersect(ray.origin, invDir, dirIsNeg, ray.distance)) {
				if (node->primitiveCount > 0) {
					// LEAF, loop through all primitives in the node to get intersection
					for (int i = 0; i < node->primitiveCount; ++i) {

						float intersection = primitives[node->primitiveOffset + i].intersect(ray.origin, ray.direction);
						if (intersection > epsilon && intersection < ray.distance && ((ray.distance - intersection) > epsilon)) {
							ray.identifier = node->primitiveOffset + i;
							ray.distance = intersection;

							hit = true;
						}
					}
					if (toVisitOffset == 0)
						break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
				} else {
					// Choose which one to visit by looking at ray direction
					if (dirIsNeg[node->splitAxis]) {
						nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
						currentNodeIndex = node->secondChildOffset;
					} else {
						nodesToVisit[toVisitOffset++] = node->secondChildOffset;
						currentNodeIndex = currentNodeIndex + 1;
					}
				}
			} else {
				if (toVisitOffset == 0)
					break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
		}
		return hit;
	}

	//Execute normal intersection but also calcualte number of ray traversals for each ray.
	__device__ bool intersect_debug(RayQueue& ray, int* traversals) {
		bool hit = false;
		glm::vec3 invDir = 1.f / ray.direction;
		int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
		// Follow ray through BVH nodes to find primitive intersections
		int toVisitOffset = 0, currentNodeIndex = 0;
		int nodesToVisit[64];
		*traversals = -1;
		while (true) {
			*traversals = *traversals + 1;
			const auto* node = &nodes[currentNodeIndex];
			// Check ray against BVH node
			if (node->bbox.intersect(ray.origin, invDir, dirIsNeg, ray.distance)) {
				if (node->primitiveCount > 0) {
					// LEAF, loop through all primitives in the node to get intersection
					for (int i = 0; i < node->primitiveCount; ++i) {

						float intersection = primitives[node->primitiveOffset + i].intersect(ray.origin, ray.direction);
						if (intersection > epsilon && intersection < ray.distance && ((ray.distance - intersection) > epsilon)) {
							ray.identifier = node->primitiveOffset + i;
							ray.distance = intersection;

							hit = true;
						}
					}
					if (toVisitOffset == 0)
						break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
				} else {
					// Choose which one to visit by looking at ray direction
					if (dirIsNeg[node->splitAxis]) {
						nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
						currentNodeIndex = node->secondChildOffset;
					} else {
						nodesToVisit[toVisitOffset++] = node->secondChildOffset;
						currentNodeIndex = currentNodeIndex + 1;
					}
				}
			} else {
				if (toVisitOffset == 0)
					break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
		}
		return hit;
	}

	/// Intersects the ray with the BVH
	/// For shadow rays we only really care if it hits anything at all
	__device__ bool intersectSimple(ShadowQueue& ray, const float& closestAllowed) {
		float closestIntersection = closestAllowed;

		glm::vec3 invDir = 1.f / ray.direction;
		int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };
		// Follow ray through BVH nodes to find primitive intersections
		int toVisitOffset = 0, currentNodeIndex = 0;
		int nodesToVisit[64];
		while (true) {
			const auto* node = &nodes[currentNodeIndex];
			// Check ray against BVH node
			if (node->bbox.intersect(ray.origin, invDir, dirIsNeg, closestIntersection)) {
				if (node->primitiveCount > 0) {
					// LEAF
					for (int i = 0; i < node->primitiveCount; ++i) {
						float intersection = primitives[node->primitiveOffset + i].intersect(ray.origin, ray.direction);
						if (intersection > epsilon && ((closestIntersection - intersection) > epsilon)) {
							//Once we've found a closer intersection than the one we accepted, return true
							//closestIntersection = intersection;
							return true;
						}
					}

					if (toVisitOffset == 0)
						break;
					currentNodeIndex = nodesToVisit[--toVisitOffset];
				} else {
					// Choose which one to visit by looking at ray direction
					if (dirIsNeg[node->splitAxis]) {
						nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
						currentNodeIndex = node->secondChildOffset;
					} else {
						nodesToVisit[toVisitOffset++] = node->secondChildOffset;
						currentNodeIndex = currentNodeIndex + 1;
					}
				}
			} else {
				if (toVisitOffset == 0)
					break;
				currentNodeIndex = nodesToVisit[--toVisitOffset];
			}
		}
		return false;
	}
};
