#include "stdafx.h"

BVH::BVH(std::vector<Triangle>& primitives, std::vector<BBox> primitivesBBoxes,
		 PartitionAlgorithm partitionAlgo)
	: partitionAlgorithm(partitionAlgo) {

	std::cout << "Creating BVH, total primitives: " << primitives.size() << "\n";
	if (primitives.size() == 0) {
		return;
	}
	nodes.resize(2 * primitives.size() - 1);

	std::vector<PrimitiveInfo> primitiveInfo(primitives.size());
	for (size_t i = 0; i < primitives.size(); ++i) {
		primitiveInfo[i] = PrimitiveInfo(i, primitivesBBoxes[i]);
	}

	//We'll hold contiguous leaves next to one another in orderedPrimitives
	std::vector<Triangle> orderedPrimitives;
	orderedPrimitives.reserve(primitives.size());

	recursiveBuild(0, primitives.size(), &nNodes, primitiveInfo,
				   orderedPrimitives, primitives);
	primitives = orderedPrimitives;

	//Output BVH statistics
	std::cout << "Created BVH, total nodes : " << nNodes << "\n";
	int dim[3] = {};
	int interior_nodes = 0;
	int leaf_nodes = 0;
	int empty = 0;
	for (int i = 0; i < nNodes; ++i) {
		if (nodes[i].primitiveCount == 0) {
			++interior_nodes;
			dim[nodes[i].splitAxis]++;
		} else { //LEAF
			++leaf_nodes;
		}
	}
	std::cout << "Interior Nodes : " << interior_nodes << "\n";
	std::cout << "Leaf Nodes : " << leaf_nodes << "\n";
	std::cout << "X-Dim : " << dim[0] << " Y-Dim : " << dim[1] << " Z-Dim : " << dim[2] << "\n";
}

int BVH::computeBucket(PrimitiveInfo primitive, glm::vec3 centroidBottom, glm::vec3 centroidTop, int dim) {
	//Get the distance from the start of the split in the axis;
	float distance = primitive.centroid[dim] - centroidBottom[dim];
	//Normalize the distance
	if (centroidTop[dim] > centroidBottom[dim]) {
		//Normalize to [0,1]
		distance = distance / (centroidTop[dim] - centroidBottom[dim]);
	}
	int bucket_idx = (int)(bucket_number * distance);
	if (bucket_idx == bucket_number) { //Can only happen if last primitive in axis has bottom == top
		bucket_idx--;
	}
	return bucket_idx;
}

//Recursively build a BVH node
void BVH::recursiveBuild(int start, int end, int* nNodes,
						 std::vector<PrimitiveInfo>& primitiveInfo,
						 std::vector<Triangle>& orderedPrimitives,
						 const std::vector<Triangle>& primitives) {
	assert(start != end && "Start == END recursive build");

	*nNodes = *nNodes + 1;
	int currentNode = *nNodes - 1;

	BBox nodeBBox = {};
	for (int i = start; i < end; ++i) {
		nodeBBox = Union(nodeBBox, primitiveInfo[i].bbox);
	}

	int nPrimitives = end - start;

	if (nPrimitives == 1) { // LEAF
		int firstPrimOffset = orderedPrimitives.size();
		int primitiveNumber = primitiveInfo[start].primitiveNumber;
		orderedPrimitives.push_back(primitives[primitiveNumber]);
		nodes[currentNode].initLeaf(firstPrimOffset, nPrimitives, nodeBBox);
		return;
	}

	/* Not leaf, get centroid bounds*/
	BBox centroidBBox;
	for (int i = start; i < end; ++i) {
		centroidBBox.addVertex(primitiveInfo[i].centroid);
	}

	/* We split based on the largest axis*/
	int dim = centroidBBox.largestExtent();

	// Partition primitives into equally-sized subsets
	int mid = (start + end) / 2;

	const glm::vec3(&centroidBottom) = centroidBBox.bounds[0];
	const glm::vec3(&centroidTop) = centroidBBox.bounds[1];

	/* Handle case of stacked bbboxes with same centroid*/
	if (centroidBottom[dim] == centroidTop[dim]) {
		// Create leaf _BVHBuildNode_
		int firstPrimOffset = orderedPrimitives.size();
		for (int i = start; i < end; ++i) {
			int primNum = primitiveInfo[i].primitiveNumber;
			orderedPrimitives.push_back(primitives[primNum]);
		}
		nodes[currentNode].initLeaf(firstPrimOffset, nPrimitives, nodeBBox);
		return;
	} else {
		// Partition primitives based on _splitMethod_
		switch (partitionAlgorithm) {
		case PartitionAlgorithm::EqualCounts: {

			std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
							 &primitiveInfo[end - 1] + 1,
							 [dim](const PrimitiveInfo& a, const PrimitiveInfo& b) {
								 return a.centroid[dim] < b.centroid[dim];
							 });
			break;
		}
		case PartitionAlgorithm::SAH: {
			//Init buckets for binned SAH
			struct Bucket {
				int count = { 0 };
				BBox bounds;
			};
			Bucket buckets[bucket_number] = {};

			//Place all the primitives in the buckets
			for (int i = start; i < end; ++i) {
				int bucket_idx = computeBucket(primitiveInfo[i], centroidBottom, centroidTop, dim);
				buckets[bucket_idx].count++;
				buckets[bucket_idx].bounds = Union(buckets[bucket_idx].bounds, primitiveInfo[i].bbox);
			}
			//Determine cost after splitting at each bucket
			//Split is [0,currentBucket] and (currentBucket, bucket_number - 2].
			//Splitting at last bucket would result in no actual split

			float cost[bucket_number] = {};
			float min_split_cost = FLT_MAX;
			int min_split_bucket = -1;
			for (int current_bucket = 0; current_bucket < bucket_number - 1; ++current_bucket) {
				int count_first_interval = 0;
				int count_second_interval = 0;
				BBox bbox_first_interval = {};
				BBox bbox_second_interval = {};

				//[0,current_bucket]
				for (int i = 0; i <= current_bucket; ++i) {
					bbox_first_interval = Union(bbox_first_interval, buckets[i].bounds);
					count_first_interval += buckets[i].count;
				}
				//(current_bucket, bucket_number - 1]
				for (int i = current_bucket + 1; i < bucket_number; ++i) {
					bbox_second_interval = Union(bbox_second_interval, buckets[i].bounds);
					count_second_interval += buckets[i].count;
				}
				//Compute SAH cost
				cost[current_bucket] = TRAVERSAL_COST + (count_first_interval * bbox_first_interval.surfaceArea() + count_second_interval * bbox_second_interval.surfaceArea()) / nodeBBox.surfaceArea();
				//Update min cost
				if (cost[current_bucket] < min_split_cost) {
					min_split_cost = cost[current_bucket];
					min_split_bucket = current_bucket;
				}
			}
			assert(min_split_bucket != -1);

			float leaf_cost = INTERSECTION_COST * nPrimitives;
			if (nPrimitives > max_prim_number || min_split_cost < leaf_cost) {
				PrimitiveInfo* partition_point = std::partition(&primitiveInfo[start],
																&primitiveInfo[end - 1] + 1,
																[=](const PrimitiveInfo& pi) {
																	int bucketIndex = computeBucket(pi, centroidBottom, centroidTop, dim);
																	return bucketIndex <= min_split_bucket;
																});
				//TODO(Dan): Is this technically undefined?
				mid = partition_point - &primitiveInfo[0];
			} else {
				int firstPrimOffset = orderedPrimitives.size();
				for (int i = start; i < end; ++i) {
					int primNum = primitiveInfo[i].primitiveNumber;
					orderedPrimitives.push_back(primitives[primNum]);
				}
				nodes[currentNode].initLeaf(firstPrimOffset, nPrimitives, nodeBBox);
				return;
			}
			break;
		}
		default: {
			std::cerr << "Error! No Valid partition algorithm selected!\n";
			break;
		}
		}
		//Nodes stored in depth first order ( exactly how nNodes grows)
		//First we build the left subtree, this will always be currentNode + 1
		recursiveBuild(start, mid, nNodes, primitiveInfo,
					   orderedPrimitives, primitives);
		//After we're done nNodes has been changed and we want to know
		//where to place the second child. That's the next free position,
		//so  nNodes.
		nodes[currentNode].secondChildOffset = *nNodes;
		recursiveBuild(mid, end, nNodes, primitiveInfo,
					   orderedPrimitives, primitives);

		//Initialize interior node
		nodes[currentNode].initInterior(dim,
										nodes[currentNode + 1],
										nodes[nodes[currentNode].secondChildOffset]);
	}
	return;
}

void BVH::BVHNode::initLeaf(int first, int n, const BBox& box) {
	bbox = box;
	primitiveOffset = first;
	primitiveCount = n;
}

void BVH::BVHNode::initInterior(int axis, const BVHNode& leftNode,
								const BVHNode& rightNode) {
	bbox = Union(leftNode.bbox, rightNode.bbox);
	primitiveCount = 0;
	splitAxis = axis;
}